# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math as th
from typing import Any, Optional, Tuple

import torch
from torch import nn
from torch.distributions import Beta
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature
import tree

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.modules.dit import AlternateVLDiT, DiT, SelfAttentionTransformer
from gr00t.model.modules.embodiment_conditioned_mlp import (
    CategorySpecificMLP,
    MultiEmbodimentActionEncoder,
)


logger = logging.getLogger(__name__)


class Gr00tN1d7ActionHead(nn.Module):
    """Action head component for flow matching diffusion policy."""

    supports_gradient_checkpointing = True

    def __init__(self, config: Gr00tN1d7Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        if config.use_alternate_vl_dit:
            self.model = AlternateVLDiT(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
                attend_text_every_n_blocks=config.attend_text_every_n_blocks,
            )
            logger.info("Using AlternateVLDiT for diffusion model")
        else:
            self.model = DiT(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
            )
            logger.info("Using DiT for diffusion model")
        self.action_dim = config.max_action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim * config.state_history_length,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=self.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )

        if config.body_action_dim is None and config.hand_action_dim is not None:
            raise ValueError("hand_action_dim requires body_action_dim to be configured")

        self.use_separate_hand_head = config.body_action_dim is not None
        if self.use_separate_hand_head:
            if not 0 < config.body_action_dim < self.action_dim:
                raise ValueError(
                    f"body_action_dim must be in [1, {self.action_dim - 1}], "
                    f"got {config.body_action_dim}"
                )
            if config.hand_action_dim is None or config.hand_action_dim <= 0:
                raise ValueError(
                    "hand_action_dim must be a positive integer when body_action_dim is set"
                )
            if config.body_action_dim + config.hand_action_dim > self.action_dim:
                raise ValueError(
                    "body_action_dim + hand_action_dim must not exceed max_action_dim, "
                    f"got {config.body_action_dim} + {config.hand_action_dim} > "
                    f"{self.action_dim}"
                )
            if not th.isfinite(config.hand_loss_weight) or config.hand_loss_weight < 0:
                raise ValueError(
                    "hand_loss_weight must be finite and non-negative, "
                    f"got {config.hand_loss_weight}"
                )
            # Do not encode the hand as another embodiment.  Body and hand use
            # independent parameters while retaining the real embodiment id.
            self.hand_action_encoder = MultiEmbodimentActionEncoder(
                action_dim=self.action_dim,
                hidden_size=self.input_embedding_dim,
                num_embodiments=config.max_num_embodiments,
            )
            self.hand_action_decoder = CategorySpecificMLP(
                num_categories=config.max_num_embodiments,
                input_dim=self.hidden_size,
                hidden_dim=self.hidden_size,
                output_dim=self.action_dim,
            )

        self.vlln = (
            nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        )

        vl_self_attention_cfg = getattr(config, "vl_self_attention_cfg", None)
        if vl_self_attention_cfg and vl_self_attention_cfg.get("num_layers", 0) > 0:
            self.vl_self_attention = SelfAttentionTransformer(**vl_self_attention_cfg)
        else:
            self.vl_self_attention = nn.Identity()

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # State dropout parameters
        self.state_dropout_prob = config.state_dropout_prob

        # Pin the time-sampling Beta to CPU/fp32 explicitly. The action head can
        # be instantiated under a meta / no_init_weights default-device context
        # (e.g. nested from_pretrained). A Beta built from bare Python floats
        # would then place its concentration tensors on the meta device (or in
        # the active default dtype, e.g. bf16). With validate_args enabled that
        # already fails here in __init__ (Beta's internal .item() check cannot
        # run on meta); even with validation off, sample_time would later raise
        # or return garbage. Explicit device/dtype here makes the sampler depend
        # only on the config, not on the construction-time device/dtype context,
        # so the noise schedule is identical across SDPA/FA2/FA4 and meta vs.
        # real-device loads. config is the canonical source for these values.
        self.beta_dist = Beta(
            torch.tensor(float(config.noise_beta_alpha), dtype=torch.float32, device="cpu"),
            torch.tensor(float(config.noise_beta_beta), dtype=torch.float32, device="cpu"),
        )
        self.num_timestep_buckets = config.num_timestep_buckets
        self.set_trainable_parameters(
            config.tune_projector, config.tune_diffusion_model, config.tune_vlln
        )

    def set_trainable_parameters(
        self, tune_projector: bool, tune_diffusion_model: bool, tune_vlln: bool
    ):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        self.tune_vlln = tune_vlln
        for p in self.parameters():
            p.requires_grad = True
        if not tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            if self.use_separate_hand_head:
                self.hand_action_encoder.requires_grad_(False)
                self.hand_action_decoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        if not tune_vlln:
            self.vlln.requires_grad_(False)
            self.vl_self_attention.requires_grad_(False)
        logger.debug(f"Tune action head projector: {self.tune_projector}")
        logger.debug(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        logger.debug(f"Tune action head vlln: {self.tune_vlln}")
        # Check if any parameters are still trainable. If not, log a warning.
        if not tune_projector and not tune_diffusion_model and not tune_vlln:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    logger.debug(f"Action head trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            logger.warning("No action head trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                if self.use_separate_hand_head:
                    self.hand_action_encoder.eval()
                    self.hand_action_decoder.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()
            if not self.tune_vlln:
                self.vlln.eval()
                self.vl_self_attention.eval()

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        sample = (1 - sample) * self.config.noise_s
        return sample

    def _action_coordinate_masks(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return broadcastable body/hand masks over the action dimension."""
        if not self.use_separate_hand_head:
            raise RuntimeError("Body/hand masks require body_action_dim to be configured")
        indices = torch.arange(actions.shape[-1], device=actions.device)
        body_mask = (indices < self.config.body_action_dim).to(actions.dtype)
        hand_start = self.config.body_action_dim
        hand_end = hand_start + self.config.hand_action_dim
        hand_mask = ((indices >= hand_start) & (indices < hand_end)).to(actions.dtype)
        return body_mask.view(1, 1, -1), hand_mask.view(1, 1, -1)

    def _encode_action_features(
        self,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        embodiment_id: torch.Tensor,
    ) -> torch.Tensor:
        """Encode one action stream, or independent body and hand token streams."""
        if not self.use_separate_hand_head:
            return self.action_encoder(actions, timesteps, embodiment_id)

        body_mask, hand_mask = self._action_coordinate_masks(actions)
        body_features = self.action_encoder(actions * body_mask, timesteps, embodiment_id)
        hand_features = self.hand_action_encoder(
            actions * hand_mask, timesteps, embodiment_id
        )
        # Layout is [body tokens (H), hand tokens (H)].
        return torch.cat((body_features, hand_features), dim=1)

    def _decode_action_velocity(
        self,
        model_output: torch.Tensor,
        action_horizon: int,
        embodiment_id: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Decode the action-token portion of the DiT output."""
        if not self.use_separate_hand_head:
            pred = self.action_decoder(model_output, embodiment_id)
            return pred[:, -action_horizon:], None, None

        action_hidden = model_output[:, -(2 * action_horizon) :]
        body_hidden = action_hidden[:, :action_horizon]
        hand_hidden = action_hidden[:, action_horizon:]
        pred_body = self.action_decoder(body_hidden, embodiment_id)
        pred_hand = self.hand_action_decoder(hand_hidden, embodiment_id)
        body_mask, hand_mask = self._action_coordinate_masks(pred_body)
        pred_body = pred_body * body_mask
        pred_hand = pred_hand * hand_mask
        return pred_body + pred_hand, pred_body, pred_hand

    def _add_action_position_embedding(
        self, action_features: torch.Tensor
    ) -> torch.Tensor:
        if not self.config.add_pos_embed:
            return action_features

        sequence_length = action_features.shape[1]
        if self.use_separate_hand_head and sequence_length % 2 != 0:
            raise ValueError(
                "Split body/hand action features must contain two equal token streams, "
                f"got sequence length {sequence_length}"
            )

        # Layout:
        # body_0 ... body_H-1, hand_0 ... hand_H-1
        # Use distinct positions for the two token streams.
        pos_ids = torch.arange(
            sequence_length,
            dtype=torch.long,
            device=action_features.device,
        )

        pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
        return action_features + pos_embs

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        backbone_features = backbone_output["backbone_features"]
        backbone_features = self.vlln(backbone_features)
        backbone_features = self.vl_self_attention(backbone_features)
        backbone_output["backbone_features"] = backbone_features
        return backbone_output

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        """
        Forward pass through the action head.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - action: [B, action_horizon, action_dim] (during training)
                - embodiment_id: [B] (embodiment IDs)
                - action_mask: [B, action_horizon, action_dim]

        Returns:
            BatchFeature containing:
                - loss: action prediction loss
        """
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features

        # Get embodiment ID.
        embodiment_id = action_input.embodiment_id

        # Handle state history
        assert action_input.state.shape[1] == self.config.state_history_length
        action_input.state = action_input.state.view(action_input.state.shape[0], 1, -1)

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Dropout state features (training only): zero out dropped states.
        if self.training and self.state_dropout_prob > 0:
            do_dropout = (
                torch.rand(state_features.shape[0], device=state_features.device)
                < self.state_dropout_prob
            )
            do_dropout = do_dropout[:, None, None].to(dtype=state_features.dtype)
            state_features = state_features * (1 - do_dropout)

        # Embed noised action trajectory.
        actions = action_input.action
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        if self.use_separate_hand_head:
            body_mask, hand_mask = self._action_coordinate_masks(actions)
            noise = noise * (body_mask + hand_mask)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self._encode_action_features(noisy_trajectory, t_discretized, embodiment_id)

        # Maybe add position embedding.
        action_features = self._add_action_position_embedding(action_features)

        # Join vision, language, state and action embedding along sequence dimension.
        sa_embs = torch.cat((state_features, action_features), dim=1)
        vl_attn_mask = backbone_output.backbone_attention_mask

        if self.config.use_alternate_vl_dit:
            image_mask = backbone_output.image_mask
            backbone_attention_mask = backbone_output.backbone_attention_mask
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized,
                return_all_hidden_states=True,
                image_mask=image_mask,
                backbone_attention_mask=backbone_attention_mask,
            )
        else:
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized,
                return_all_hidden_states=True,
            )

        pred_actions, _, _ = self._decode_action_velocity(
            model_output,
            actions.shape[1],
            embodiment_id,
        )

        # Slice out only the action portion of pred and target.
        action_mask = action_input.action_mask
        action_loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
        if not self.use_separate_hand_head:
            loss = action_loss.sum() / (action_mask.sum() + 1e-6)
            body_loss = None
            hand_loss = None
        else:
            body_action_dim = self.config.body_action_dim
            hand_action_end = body_action_dim + self.config.hand_action_dim
            body_mask = action_mask[..., :body_action_dim]
            hand_mask = action_mask[..., body_action_dim:hand_action_end]
            body_loss_sum = action_loss[..., :body_action_dim].sum()
            hand_loss_sum = action_loss[..., body_action_dim:hand_action_end].sum()
            body_loss_count = body_mask.sum()
            hand_loss_count = hand_mask.sum()
            body_loss = body_loss_sum / body_loss_count.clamp_min(1e-6)
            hand_loss = hand_loss_sum / hand_loss_count.clamp_min(1e-6)
            loss = body_loss + self.config.hand_loss_weight * hand_loss

        outputs = {
            "loss": loss,
            "action_loss": action_loss,
            "action_mask": action_mask,
            "backbone_features": vl_embeds,
            "state_features": state_features,
        }
        if body_loss is not None and hand_loss is not None:
            outputs["body_loss"] = body_loss.detach()
            outputs["hand_loss"] = hand_loss.detach()

        return outputs

    def _encode_features(
        self, backbone_output: BatchFeature, action_input: BatchFeature
    ) -> BatchFeature:
        """
        Encode features for the action head.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_history_length, max_state_dim]
                - embodiment_id: [B] (embodiment IDs)

        Returns:
            BatchFeature containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - state_features: [B, 1, input_embedding_dim]
        """
        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id

        # Handle state history: if we have fewer timesteps than expected, repeat to fill
        state = action_input.state
        current_T = state.shape[1]
        assert current_T == self.config.state_history_length, "current_T != state_history_length"
        # Reshape state from [B, state_history_length, max_state_dim] to [B, 1, state_history_length * max_state_dim]
        state = state.view(state.shape[0], 1, -1)

        # Embed state.
        state_features = self.state_encoder(state, embodiment_id)

        return BatchFeature(data={"backbone_features": vl_embeds, "state_features": state_features})

    @torch.no_grad()
    def get_action_with_features(
        self,
        backbone_features: torch.Tensor,
        state_features: torch.Tensor,
        embodiment_id: torch.Tensor,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        options: dict[str, Any] | None = None,
    ) -> BatchFeature:
        """
        Generate actions using the flow matching diffusion process.

        Args:
            backbone_features: [B, seq_len, backbone_embedding_dim]
            state_features: [B, state_horizon, input_embedding_dim]
            embodiment_id: [B] (embodiment IDs)
            backbone_output: Output from the backbone model
        """
        vl_embeds = backbone_features

        # Set initial actions as the sampled noise.
        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.action_dim),
            dtype=vl_embeds.dtype,
            device=device,
        )
        if self.use_separate_hand_head:
            print("use_separate_hand_head!!!!!")
            body_mask, hand_mask = self._action_coordinate_masks(actions)
            actions = actions * (body_mask + hand_mask)

        dt = 1.0 / self.num_inference_timesteps
        vel_strength = torch.ones_like(actions)

        rtc_keys = {"rtc_overlap_steps", "rtc_frozen_steps", "rtc_ramp_rate"}
        rtc_option_keys = rtc_keys | {"rtc_prev_action"}
        provided_rtc_option_keys = rtc_option_keys.intersection(options or {})
        if provided_rtc_option_keys:
            # rtc_overlap_steps is the number of steps to overlap with the previous action chunks.
            # rtc_frozen_steps is the number of steps to freeze the action, which is the latency of the policy inference.
            # rtc_ramp_rate is the rate of the ramp of denoising the actions.
            missing_rtc_keys = sorted(rtc_keys - options.keys())
            if missing_rtc_keys:
                raise ValueError(f"Missing GR00T RTC options: {missing_rtc_keys}")

            rtc_overlap_steps = int(options["rtc_overlap_steps"])
            rtc_frozen_steps = int(options["rtc_frozen_steps"])
            rtc_ramp_rate = float(options["rtc_ramp_rate"])
            if not 0 <= rtc_frozen_steps <= rtc_overlap_steps <= self.action_horizon:
                raise ValueError(
                    "GR00T RTC requires 0 <= rtc_frozen_steps <= rtc_overlap_steps "
                    f"<= action_horizon, got frozen={rtc_frozen_steps}, "
                    f"overlap={rtc_overlap_steps}, H={self.action_horizon}"
                )
            if not th.isfinite(rtc_ramp_rate) or rtc_ramp_rate <= 0:
                raise ValueError(f"rtc_ramp_rate must be positive, got {rtc_ramp_rate}")

            if "action" in action_input:
                previous_actions = torch.as_tensor(
                    action_input["action"],
                    device=device,
                    dtype=actions.dtype,
                )
                if previous_actions.shape != actions.shape:
                    raise ValueError(
                        f"GR00T RTC previous action shape error: expected {tuple(actions.shape)}, "
                        f"got {tuple(previous_actions.shape)}"
                    )
                if not torch.isfinite(previous_actions).all():
                    raise ValueError("GR00T RTC previous action must contain only finite values")

                # Use previous action instead of pure noise to do inpainting
                if rtc_overlap_steps > 0:
                    actions[:, :rtc_overlap_steps, :] = previous_actions[
                        :,
                        -rtc_overlap_steps:,
                        :,
                    ]
                vel_strength[:, :rtc_frozen_steps, :] = 0.0
                # NOTE: use an exponential ramp strength to set the remaining unfrozen rtc_steps
                intermediate_steps = rtc_overlap_steps - rtc_frozen_steps
                # Create exponential ramp from 0 to 1 over intermediate steps
                t = torch.linspace(0.0, 1.0, intermediate_steps + 2, device=device)
                ramp = 1 - torch.exp(-rtc_ramp_rate * t)
                ramp = ramp / ramp[-1].clamp_min(1e-8)  # normalize to [0,1]
                ramp = ramp[
                    1:-1
                ]  # we will only take the middle part of the ramp, ignore the 0.0 and 1.0
                # Apply ramp to the intermediate steps [batch, intermediate_steps, action_dim]
                vel_strength[
                    :,
                    rtc_frozen_steps:rtc_overlap_steps,
                    :,
                ] = ramp[None, :, None].to(device)

        if self.use_separate_hand_head:
            body_mask, hand_mask = self._action_coordinate_masks(actions)
            actions = actions * (body_mask + hand_mask)

        # Run denoising steps.
        for t in range(self.num_inference_timesteps):
            t_cont = t / float(self.num_inference_timesteps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Embed noised action trajectory.
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            action_features = self._encode_action_features(
                actions, timesteps_tensor, embodiment_id
            )
            action_features = self._add_action_position_embedding(action_features)

            # Join vision, language, state and action embedding along sequence dimension.
            sa_embs = torch.cat((state_features, action_features), dim=1)

            # Run model forward.
            if self.config.use_alternate_vl_dit:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                    image_mask=backbone_output.image_mask,
                    backbone_attention_mask=backbone_output.backbone_attention_mask,
                )
            else:
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                )
            pred_velocity, _, _ = self._decode_action_velocity(
                model_output, self.action_horizon, embodiment_id
            )

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity * vel_strength

        return BatchFeature(
            data={
                "action_pred": actions,
                "backbone_features": vl_embeds,
                "state_features": state_features,
            }
        )

    @torch.no_grad()
    def get_action(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        options: dict[str, Any] | None = None,
    ) -> BatchFeature:
        """
        Generate actions using the flow matching diffusion process.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - embodiment_id: [B] (embodiment IDs)

        Returns:
            BatchFeature containing:
                - action_pred: [B, action_horizon, action_dim] predicted actions
        """
        features = self._encode_features(backbone_output, action_input)
        return self.get_action_with_features(
            backbone_features=features.backbone_features,
            state_features=features.state_features,
            embodiment_id=action_input.embodiment_id,
            backbone_output=backbone_output,
            action_input=action_input,
            options=options,
        )

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

    def prepare_input(self, batch: dict) -> BatchFeature:
        """Prepare input batch for the action head."""
        return BatchFeature(data=batch)


def get_backbone_cls(config: Gr00tN1d7Config):
    if (
        config.backbone_model_type == "qwen"
        or "nvidia/Cosmos-Reason2" in config.model_name
        or "Qwen/Qwen3-VL" in config.model_name
    ):
        # We import here as Qwen3Backbone depends on newer transformers versions than the rest of the code.
        from gr00t.model.modules.qwen3_backbone import Qwen3Backbone

        return Qwen3Backbone
    else:
        raise ValueError(f"Unsupported model name: {config.model_name}")


class Gr00tN1d7(PreTrainedModel):
    """Gr00tN1d7: VLA model with Cosmos-Reason2-2B (Qwen3-VL) backbone."""

    config_class = Gr00tN1d7Config
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: Gr00tN1d7Config,
        transformers_loading_kwargs: dict = {"trust_remote_code": True},
    ):
        """
        Initialize Gr00tN1d7 model.

        Args:
            config: Model configuration
            transformers_loading_kwargs: Dict with transformers loading parameters:
                - transformers_trust_remote_code: Whether to trust remote code when loading from HF Hub
                - transformers_local_files_only: Whether to only use local files
                - model_revision: Specific model revision to use
                - transformers_cache_dir: Directory to cache downloaded models
                - transformers_access_token: HuggingFace access token for gated models

        Note: During training, transformers parameters are passed from training config.
              During inference (e.g., from_pretrained), defaults are used.
        """
        super().__init__(config)
        self.config = config

        backbone_cls = get_backbone_cls(config)
        self.backbone = backbone_cls(
            model_name=config.model_name,
            tune_llm=config.tune_llm,
            tune_visual=config.tune_visual,
            select_layer=config.select_layer,
            reproject_vision=config.reproject_vision,
            use_flash_attention=config.use_flash_attention,
            load_bf16=config.load_bf16,
            tune_top_llm_layers=config.tune_top_llm_layers,
            trainable_params_fp32=config.backbone_trainable_params_fp32,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )

        # Initialize action head
        self.action_head = Gr00tN1d7ActionHead(config)
        from .processing_gr00t_n1d7 import Gr00tN1d7DataCollator

        self.collator = Gr00tN1d7DataCollator(
            model_name=config.model_name,
            model_type=config.backbone_model_type,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )

    def prepare_input(self, inputs: dict) -> Tuple[BatchFeature, BatchFeature]:
        """Prepare inputs for backbone and action head."""

        # NOTE -- currently the eval code doesn't use collator, so we need to add it here
        # this should ideally be fixed upstream
        if "vlm_content" in inputs:
            # Fix for n_envs > 1: Process all environments' VLM content, not just the first
            vlm_content_list = inputs["vlm_content"]
            # Ensure vlm_content_list is always a list for consistent processing
            if not isinstance(vlm_content_list, list):
                vlm_content_list = [vlm_content_list]

            # Process all VLM contents through the collator
            prep = self.collator([{"vlm_content": vlm} for vlm in vlm_content_list])["inputs"]
            inputs.pop("vlm_content")
            inputs.update(prep)

        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

        # Move to device and dtype
        def to_device_with_dtype(x):
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.dtype)
            else:
                return x.to(self.device)

        backbone_inputs = tree.map_structure(to_device_with_dtype, backbone_inputs)
        action_inputs = tree.map_structure(to_device_with_dtype, action_inputs)

        return backbone_inputs, action_inputs

    def forward(self, inputs: dict) -> BatchFeature:
        """
        Forward pass through the complete model.

        Args:
            inputs: Dictionary containing:
                - Action inputs (state, action, embodiment_id, etc.)

        Returns:
            BatchFeature containing loss and other outputs
        """
        # Prepare inputs for backbone and action head
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_outputs = self.action_head(backbone_outputs, action_inputs)

        return action_outputs

    def get_action(self, inputs: dict, options: dict[str, Any] | None = None) -> BatchFeature:
        """
        Generate actions using the complete model.
        """
        # Prepare inputs for backbone and action head
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        if options is not None and options.get("rtc_prev_action") is not None:
            action_inputs["action"] = options["rtc_prev_action"]
        # Forward through backbone
        backbone_outputs = self.backbone(backbone_inputs)
        action_outputs = self.action_head.get_action(backbone_outputs, action_inputs, options)

        return action_outputs

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


# Register the model with HuggingFace
AutoConfig.register("Gr00tN1d7", Gr00tN1d7Config)
AutoModel.register(Gr00tN1d7Config, Gr00tN1d7)

#!/usr/bin/env python3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path
import random
import shutil

from convert_to_lerobot_with_hand_new import (
    OUTPUT_JSON_NAME,
    DataSetArgs,
    LeRobotDataset,
    _collect_episodes,
    _finalize_video_only_dataset,
    _load_frame_images,
    load_windowed_episode,
)
import numpy as np
import tyro


def _collect_sampled_episodes(
    input_dirs: list[Path], sampling_rate: float, sampling_seed: int
) -> list[Path]:
    """Sample complete episodes independently from each input directory."""
    rng = random.Random(sampling_seed)
    sampled_episodes: list[Path] = []

    for input_dir in input_dirs:
        episodes = _collect_episodes([input_dir])
        if not episodes:
            continue

        sample_count = max(1, int(len(episodes) * sampling_rate))
        selected = sorted(rng.sample(episodes, sample_count))
        print(
            f"{input_dir}: sampled {len(selected)}/{len(episodes)} episode(s) "
            f"with rate={sampling_rate}"
        )
        sampled_episodes.extend(selected)

    return sampled_episodes


def convert_sampled(args: "SamplingDataSetArgs", episode_dirs: list[Path]) -> None:
    """Convert a preselected list of episodes without changing the original converter."""
    print(f"Total: {len(episode_dirs)} sampled episode(s)\n")

    output_path = Path(args.output_dir)
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        robot_type=args.robot_type,
        fps=args.fps,
        features=args.features,
        root=output_path,
        image_writer_threads=args.image_writer_threads,
    )
    dataset.meta.info["num_state_action_per_frame"] = args.interval_num + 1

    for ep_idx, ep_dir in enumerate(episode_dirs):
        print(f"[{ep_idx + 1}/{len(episode_dirs)}] {ep_dir}")
        windowed_frames = load_windowed_episode(ep_dir, args)

        if args.save_window_json:
            json_frames = [
                {
                    **frame,
                    "states": frame["states"].tolist(),
                    "action": frame["action"].tolist(),
                }
                for frame in windowed_frames
            ]
            with (ep_dir / OUTPUT_JSON_NAME).open("w", encoding="utf-8") as f:
                json.dump(json_frames, f, ensure_ascii=False, indent=2)

        image_size = (args.image_shape[1], args.image_shape[0])
        instruction = None
        with ThreadPoolExecutor(max_workers=args.image_loader_threads) as loader:
            img_futures = [
                loader.submit(
                    _load_frame_images,
                    ep_dir,
                    frame,
                    args.placeholder_image,
                    image_size,
                )
                for frame in windowed_frames
            ]

            for frame, img_future in zip(windowed_frames, img_futures):
                head_img, left_wrist_img, right_wrist_img = img_future.result()
                states = frame.get("states")
                action = frame.get("action")
                task = frame.get("task")

                assert states is not None, f"states missing in {ep_dir}"
                assert action is not None, f"action missing in {ep_dir}"
                assert task is not None and isinstance(task, list), (
                    f"task missing or invalid in {ep_dir}"
                )
                instruction = task[0]

                formatted_frame = {
                    "observation.state": states,
                    "observation.images.head_img": head_img,
                    "observation.images.left_wrist_img": left_wrist_img,
                    "observation.images.right_wrist_img": right_wrist_img,
                    "action": action,
                    "task_vis_stickman": np.zeros((900,), dtype=np.float64),
                }
                dataset.add_frame(formatted_frame)

        dataset.save_episode(task=instruction)

    _finalize_video_only_dataset(
        output_path=output_path,
        cameras=args.cameras,
        num_episodes=len(episode_dirs),
    )
    print(f"\nDone! {len(episode_dirs)} sampled episodes saved to {output_path}")


@dataclass
class SamplingDataSetArgs(DataSetArgs):
    sampling_rate: float = 1.0
    """Fraction of complete episodes sampled independently from each input directory."""

    sampling_seed: int = 42
    """Random seed used for reproducible episode sampling."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0.0 < self.sampling_rate <= 1.0:
            raise ValueError("sampling_rate must be in the range (0, 1].")


if __name__ == "__main__":
    args = tyro.cli(SamplingDataSetArgs)
    if not args.input_dirs:
        raise ValueError("Provide at least one input directory via --input-dirs.")

    episode_dirs = _collect_sampled_episodes(
        input_dirs=args.input_dirs,
        sampling_rate=args.sampling_rate,
        sampling_seed=args.sampling_seed,
    )
    if not episode_dirs:
        raise ValueError("No valid episodes were found in the input directories.")

    convert_sampled(args, episode_dirs)

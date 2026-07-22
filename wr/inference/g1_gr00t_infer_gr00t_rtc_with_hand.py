import os
import sys
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from time import sleep
from typing import Any, Callable

import numpy as np
import tyro
from loop_rate_limiters import RateLimiter
from PIL import Image

sys.path.append("/home/unitree/liyifan/wr/")

from data_res.camera import VideoCapture, CameraGrabber
from data_res.dds import (
    MOCAP_NUM_JOINTS,
    MOCAP_POS_DIM,
    MOCAP_QUAT_DIM,
    BodyPoseConfig,
    BodyPoseSubscriberV3,
    MocapConfig,
    MocapUE5G115MsgPublisher,
    MocapUEHandPublisher,
    MocapUEHandSubscriber,
    WR_GAE_BodyPose_Msg_V2,
)
from data_res.log import get_logger
from data_res.transforms import (
    compute_absolute,
    compute_imu_relative,
    compute_relative,
    interpolate_pose7,
    normalize_quaternion,
    quaternion_to_rotation_6d,
    restore_mocap_from_root_relative_delta,
    rotation_6d_to_quaternion,
)
from data_res.utils import CAMERAS_MAP, SELECT_11_INDICES, get_camera_name
from serve_res.gr00t import server_client

logger = get_logger(__name__)


@dataclass
class ClientConfig:
    host: str = "192.168.123.163"
    port: int = 9002
    timeout_ms: int = 15000
    api_token: str | None = None
    task_description: str = (
        "pick up the cube and bottle into the bowl"
    )
    send_fps: float = 50.0
    history_len: int = 50
    camera_flush_infer: int = 5
    rtc_init_delay_steps: int = 5
    rtc_delay_buffer_len: int = 3
    rtc_ramp_rate: float = 1.0
    fallback_sync_on_async_error: bool = True
    enable_action_interp: bool = False
    interp_num: int = 1
    rtc_submit_remaining_frames: int = 20
    root_pose_z: float | None = 1.0
    mocap_cfg: MocapConfig = field(
        default_factory=lambda: MocapConfig(
            domain_id=1, topic_name="MocapUE5G115Topic", depth=4
        )
    )
    body_pose_cfg: BodyPoseConfig = field(
        default_factory=lambda: BodyPoseConfig(
            domain_id=1, topic_name="WR/BodyPose", depth=4
        )
    )
    camera_config: Path = Path("config/camera.yaml")


@dataclass
class InferenceResult:
    actions_abs: np.ndarray
    hands_abs: np.ndarray
    actions_rel: np.ndarray
    chunk_start_root_rel: np.ndarray
    chunk_start_joint_rel: np.ndarray
    rtc_overlap_steps: int | None = None
    rtc_frozen_steps: int | None = None
    submit_action_frame_count: int | None = None
    parent_chunk_id: int | None = None
    measured_delay_steps: int | None = None


@dataclass
class ActiveChunk:
    chunk_id: int
    result: InferenceResult
    frames: np.ndarray
    hand_frames: np.ndarray
    action_frame_offset: int
    prepended_bridge_frames: int
    cursor: int = 0
    rtc_submitted: bool = False

    @property
    def done(self) -> bool:
        return self.cursor >= self.frames.shape[0]

    def next_frame(self) -> tuple[np.ndarray, np.ndarray]:
        if self.done:
            raise RuntimeError("Active chunk has no remaining frames")
        return self.frames[self.cursor], self.hand_frames[self.cursor]

    def mark_sent(self) -> bool:
        is_action_frame = self.cursor >= self.prepended_bridge_frames
        self.cursor += 1
        return is_action_frame

    def sent_action_frames(self) -> int:
        return max(0, self.cursor - self.prepended_bridge_frames)

    def full_frame_count(self) -> int:
        return self.action_frame_offset + self.sent_action_frames()

    def executed_steps(self, interp_stride: int) -> int:
        return model_steps_from_frame_count(self.full_frame_count(), interp_stride)

    def at_model_boundary(self, interp_stride: int) -> bool:
        frame_count = self.full_frame_count()
        return frame_count > 0 and (frame_count - 1) % interp_stride == 0

    def root_at_step(self, action_step: int) -> np.ndarray:
        action_step = max(0, min(action_step, self.result.actions_rel.shape[0]))
        root_delta = self.result.actions_rel[:action_step, :3].sum(axis=0)
        return self.result.chunk_start_root_rel + root_delta

    def joint_relative_at_step(self, action_step: int) -> np.ndarray:
        action_step = max(0, min(action_step, self.result.actions_rel.shape[0]))
        joint_delta = self.result.actions_rel[:action_step, 3:36].reshape(-1, 11, 3)
        return self.result.chunk_start_joint_rel + joint_delta.sum(axis=0)

    def current_root(self, interp_stride: int) -> np.ndarray:
        return self.root_at_step(self.executed_steps(interp_stride))


class DelayBuffer:
    def __init__(self, initial_delay_steps: int, maxlen: int):
        if maxlen <= 0:
            raise ValueError(f"maxlen must be positive, got {maxlen}")
        self._queue = deque([max(int(initial_delay_steps), 0)], maxlen=maxlen)

    def add(self, delay_steps: int) -> None:
        self._queue.append(max(int(delay_steps), 0))

    def max(self) -> int:
        return max(self._queue)


class StateHistoryQueue:
    def __init__(self, maxlen: int):
        self._lock = threading.Lock()
        self._queue = deque(maxlen=maxlen)

    def put(self, msg: WR_GAE_BodyPose_Msg_V2, hand_state: np.ndarray) -> None:
        with self._lock:
            q_np = np.asarray(msg.robot_qpos[:29], dtype=np.float32)
            imu_np = np.asarray(msg.robot_rootquat, dtype=np.float32)
            imu_np = compute_imu_relative(imu_np[None, :], imu_np[None, :])
            imu_np = quaternion_to_rotation_6d(imu_np)[0]
            self._queue.append(
                np.concatenate([imu_np, q_np, np.asarray(hand_state, dtype=np.float32)])
            )

    def get_all(self) -> np.ndarray:
        with self._lock:
            if not self._queue:
                raise RuntimeError("History queue is empty")
            return self._copy_all_locked()

    def _copy_all_locked(self) -> np.ndarray:
        states = [state.copy() for state in self._queue]
        if len(states) < self._queue.maxlen:
            states = [states[0].copy()] * (
                self._queue.maxlen - len(states)
            ) + states
        return np.stack(states, axis=0)


class AsyncInferenceWorker:
    def __init__(self, infer_fn: Callable[..., Any], config: ClientConfig):
        self._infer_fn = infer_fn
        self._config = config
        self._lock = threading.Lock()
        self._busy = False
        self._done = threading.Event()
        self._result = None
        self._error = None

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._busy

    def submit(self, **kwargs: Any) -> bool:
        with self._lock:
            if self._busy:
                return False
            self._busy = True
            self._done.clear()
            self._result = None
            self._error = None
        threading.Thread(target=self._run, kwargs=kwargs, daemon=True).start()
        return True

    def _make_client(self):
        return server_client.PolicyClient(
            host=self._config.host,
            port=self._config.port,
            timeout_ms=self._config.timeout_ms,
            api_token=self._config.api_token,
        )

    def _run(self, **kwargs: Any) -> None:
        try:
            client = self._make_client()
            result = self._infer_fn(policy_client=client, **kwargs)
            with self._lock:
                self._result = result
        except BaseException as exc:
            with self._lock:
                self._error = exc
        finally:
            with self._lock:
                self._busy = False
            self._done.set()

    def poll_done(self) -> bool:
        with self._lock:
            return (not self._busy) and self._done.is_set() and (
                self._result is not None or self._error is not None
            )

    def get_result(self):
        with self._lock:
            error = self._error
            result = self._result
            self._done.clear()
            self._error = None
            self._result = None

        if error is not None:
            raise error
        if result is None:
            raise RuntimeError("inference finished without result")
        return result


def split_xyz_wxyz(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    expected_shape = (MOCAP_NUM_JOINTS, MOCAP_POS_DIM + MOCAP_QUAT_DIM)
    frame = np.asarray(frame, dtype=np.float32)
    if frame.shape != expected_shape:
        raise ValueError(
            f"Expected mocap frame shape {expected_shape}, got {frame.shape}"
        )
    xyz = frame[:, :MOCAP_POS_DIM]
    wxyz = normalize_quaternion(frame[:, MOCAP_POS_DIM:])
    if np.any(np.linalg.norm(wxyz, axis=-1) < 0.5):
        raise ValueError("Refusing to send mocap action with invalid quaternion")
    return xyz, wxyz


def build_action_frames(
    prev_last_pose7: np.ndarray | None,
    prev_last_hand: np.ndarray | None,
    cur_chunk_pose7: np.ndarray,
    cur_chunk_hand: np.ndarray,
    enable_interp: bool,
    allow_bridge: bool,
    num_interp: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    cur_chunk_pose7 = np.asarray(cur_chunk_pose7, dtype=np.float32)
    cur_chunk_hand = np.asarray(cur_chunk_hand, dtype=np.float32)
    if cur_chunk_pose7.ndim != 3 or cur_chunk_pose7.shape[1:] != (15, 7):
        raise ValueError(
            f"Expected cur_chunk_pose7 shape (T,15,7), got {cur_chunk_pose7.shape}"
        )
    if cur_chunk_pose7.shape[0] == 0:
        raise ValueError("Cannot install empty action chunk")
    if not enable_interp or num_interp <= 0:
        return cur_chunk_pose7, cur_chunk_hand, 0

    prepended_bridge_frames = 0
    source_pose7 = cur_chunk_pose7
    source_hand = cur_chunk_hand
    if allow_bridge and prev_last_pose7 is not None and prev_last_hand is not None:
        prev_last_pose7 = np.asarray(prev_last_pose7, dtype=np.float32)
        source_pose7 = np.concatenate(
            [prev_last_pose7[None, ...], cur_chunk_pose7],
            axis=0,
        )
        source_hand = np.concatenate(
            [np.asarray(prev_last_hand, dtype=np.float32)[None, ...], cur_chunk_hand],
            axis=0,
        )
        prepended_bridge_frames = num_interp

    frames = interpolate_pose7(source_pose7, num_interp=num_interp)
    hand_frames = interpolate_hand_joint(source_hand, num_interp=num_interp)
    if prepended_bridge_frames > 0:
        frames = frames[1:]
        hand_frames = hand_frames[1:]

    logger.info(
        "action interpolation enabled: bridge=%d, model_steps=%d, frames=%d",
        prepended_bridge_frames,
        cur_chunk_pose7.shape[0],
        frames.shape[0],
    )
    return frames.astype(np.float32), hand_frames.astype(np.float32), prepended_bridge_frames


def interpolate_hand_joint(hand_joint_seq: np.ndarray, num_interp: int) -> np.ndarray:
    hand_joint_seq = np.asarray(hand_joint_seq, dtype=np.float32)
    if hand_joint_seq.shape[0] <= 1 or num_interp <= 0:
        return hand_joint_seq
    steps = np.linspace(0.0, 1.0, num_interp + 2, dtype=np.float32)[:-1]
    out = (
        (1.0 - steps[None, :, None]) * hand_joint_seq[:-1, None, :]
        + steps[None, :, None] * hand_joint_seq[1:, None, :]
    )
    return np.concatenate([out.reshape(-1, hand_joint_seq.shape[1]), hand_joint_seq[-1:]])


def load_image_from_array(
    image_input: np.ndarray,
    target_size: tuple[int, int] = (256, 256),
) -> np.ndarray:
    image_arr = np.asarray(image_input)
    if image_arr.ndim != 3 or image_arr.shape[-1] != 3:
        raise ValueError(f"Expected frame shape (H, W, 3), got {image_arr.shape}")
    image = Image.fromarray(image_arr.astype(np.uint8)[:, :, ::-1]).convert("RGB")

    return np.asarray(image.resize(target_size, Image.BILINEAR))[None, None, ...]


def build_observation_from_msg_with_history(
    history_state: np.ndarray,
    task_description: str,
    frame: dict[str, np.ndarray] | None,
) -> dict:
    history_state = np.asarray(history_state, dtype=np.float32)
    video = {
        "ego_view": np.zeros((1, 1, 256, 256, 3), dtype=np.uint8),
        "left_wrist_view": np.zeros((1, 1, 256, 256, 3), dtype=np.uint8),
        "right_wrist_view": np.zeros((1, 1, 256, 256, 3), dtype=np.uint8),
    }
    if frame is not None:
        for name, image in frame.items():
            video[CAMERAS_MAP[name]] = load_image_from_array(
                np.asarray(image),
                (256, 256),
            )

    state = history_state.reshape(-1)
    return {
        "video": video,
        "state": {"imu_joints": state[None, None, :].astype(np.float32)},
        "language": {
            "annotation.human.task_description": [[task_description]]
        },
        "stickman": {
            "annotation.human.stickman": np.zeros(
                (1, 1, 900), dtype=np.float32
            )
        },
    }


def read_camera_frames(
    camera_caps: dict[str, VideoCapture],
    flush_count: int,
) -> dict[str, np.ndarray]:
    frames = {}
    for name, camera_cap in camera_caps.items():
        frame = None
        ok = False
        for _ in range(flush_count):
            ok, frame = camera_cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read camera frame from {name}")
        frames[name] = frame
    return frames


def wait_for_body_pose_msg(
    body_pose: BodyPoseSubscriberV3,
    poll_interval_s: float = 0.01,
) -> WR_GAE_BodyPose_Msg_V2:
    while True:
        msg = body_pose.get_msg()
        if msg is not None:
            return msg
        sleep(poll_interval_s)


def wait_for_hand_state_msg(
    hand_subscriber: MocapUEHandSubscriber,
    poll_interval_s: float = 0.01,
) -> np.ndarray:
    while True:
        state = hand_subscriber.get_state()
        if state is not None:
            return state
        sleep(poll_interval_s)


def model_steps_from_frame_count(frame_count: int, interp_stride: int) -> int:
    if frame_count <= 0:
        return 0
    return (frame_count + interp_stride - 1) // interp_stride


def action_frame_offset_after_steps(steps: int, interp_stride: int) -> int:
    if steps <= 0:
        return 0
    return (steps - 1) * interp_stride + 1


def model_action_to_abs_action(
    action_output: np.ndarray,
    init_pose: np.ndarray,
    chunk_start_root_rel: np.ndarray,
    chunk_start_joint_rel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    action_output = np.asarray(action_output, dtype=np.float32).reshape(-1, 114)
    if not np.isfinite(action_output).all():
        raise ValueError("Model returned non-finite actions")
    if not np.any(np.abs(action_output) > 1e-6):
        logger.warning("Model returned an all-zero action chunk")

    root_delta = action_output[:, :3]
    action_without_root_delta = action_output[:, 3:102]
    action_hand = action_output[:, 102:]
    action_mocap_xyz = action_without_root_delta[:, :33].reshape(-1, 11, 3)
    action_mocap_6d = action_without_root_delta[:, 33:].reshape(-1, 11, 6)
    action_mocap = np.concatenate(
        [action_mocap_xyz, action_mocap_6d],
        axis=-1,
    ).reshape(-1, 99)
    action_mocap = np.concatenate([root_delta, action_mocap], axis=-1)
    action_11x9, next_root_rel, next_joint_rel = restore_mocap_from_root_relative_delta(
        action_mocap,
        chunk_start_root_rel,
        chunk_start_joint_rel,
    )

    num_frames = action_11x9.shape[0]
    action_15x9 = np.zeros((num_frames, 15, 9), dtype=np.float32)
    action_15x9[..., 3:9] = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    action_15x9[:, SELECT_11_INDICES, :] = action_11x9

    action_15x7 = rotation_6d_to_quaternion(action_15x9)
    num_frames, num_joints, num_poses = action_15x7.shape
    action_15x7_flat = action_15x7.reshape(num_frames * num_joints, num_poses)
    init_pose_x7_flat = np.tile(init_pose, (num_frames * num_joints, 1))
    action_abs = compute_absolute(init_pose_x7_flat, action_15x7_flat).reshape(
        num_frames,
        num_joints,
        num_poses,
    )
    if (
        not np.isfinite(action_abs).all()
        or not np.isfinite(next_root_rel).all()
        or not np.isfinite(next_joint_rel).all()
    ):
        raise ValueError("Restored mocap action contains non-finite values")
    return action_abs, action_hand


def validate_config(config: ClientConfig) -> None:
    if config.send_fps <= 0:
        raise ValueError(f"send_fps must be positive, got {config.send_fps}")
    if config.history_len <= 0:
        raise ValueError(f"history_len must be positive, got {config.history_len}")
    if config.camera_flush_infer <= 0:
        raise ValueError(
            f"camera_flush_infer must be positive, got {config.camera_flush_infer}"
        )
    if config.interp_num < 0:
        raise ValueError(f"interp_num must be non-negative, got {config.interp_num}")
    if config.rtc_init_delay_steps < 0:
        raise ValueError(
            "rtc_init_delay_steps must be non-negative, "
            f"got {config.rtc_init_delay_steps}"
        )
    if config.rtc_delay_buffer_len <= 0:
        raise ValueError(
            "rtc_delay_buffer_len must be positive, "
            f"got {config.rtc_delay_buffer_len}"
        )
    if not np.isfinite(config.rtc_ramp_rate) or config.rtc_ramp_rate <= 0:
        raise ValueError(
            f"rtc_ramp_rate must be positive, got {config.rtc_ramp_rate}"
        )
    if config.rtc_submit_remaining_frames <= 0:
        raise ValueError(
            "rtc_submit_remaining_frames must be positive, "
            f"got {config.rtc_submit_remaining_frames}"
        )


def gr00t_rtc_inputs(
    config: ClientConfig,
    action_horizon: int,
    overlap_steps: int,
    frozen_steps: int,
) -> dict[str, np.ndarray]:
    if not 0 <= frozen_steps <= overlap_steps <= action_horizon:
        raise ValueError(
            "GR00T RTC requires 0 <= frozen <= overlap <= action_horizon, "
            f"got frozen={frozen_steps}, overlap={overlap_steps}, H={action_horizon}"
        )
    return {
        "rtc_overlap_steps": np.asarray(overlap_steps, dtype=np.int32),
        "rtc_frozen_steps": np.asarray(frozen_steps, dtype=np.int32),
        "rtc_ramp_rate": np.asarray(config.rtc_ramp_rate, dtype=np.float32),
    }


class Gr00tRtcV2Runner:
    def __init__(self, config: ClientConfig):
        self.config = config
        self.interp_stride = (
            config.interp_num + 1 if config.enable_action_interp else 1
        )
        self.rate = RateLimiter(frequency=config.send_fps)

        self.body_pose = BodyPoseSubscriberV3(config.body_pose_cfg)
        self.publisher = MocapUE5G115MsgPublisher(config.mocap_cfg, fps=config.send_fps)
        self.hand_subscriber = MocapUEHandSubscriber()
        self.hand_publisher = MocapUEHandPublisher()
        self.history = StateHistoryQueue(maxlen=config.history_len)
        self.delay_buffer = DelayBuffer(
            initial_delay_steps=config.rtc_init_delay_steps,
            maxlen=config.rtc_delay_buffer_len,
        )

        self.camera_caps: dict[str, VideoCapture] = {}
        self.camera_grabber: CameraGrabber | None = None
        self.main_client = None
        self.worker = AsyncInferenceWorker(self._infer_once, config)

        self.action_horizon = 30
        self.root_pose: np.ndarray | None = None
        self.root_rel_cum = np.zeros(3, dtype=np.float32)
        self.joint_rel_cum = np.zeros((11, 3), dtype=np.float32)
        self.active: ActiveChunk | None = None
        self.pending_rtc_result: InferenceResult | None = None
        self.last_sent_pose: np.ndarray | None = None
        self.last_sent_hand: np.ndarray | None = None
        self.next_chunk_id = 0

        self._sent_action_frame_count = 0
        self._counter_lock = threading.Lock()

    def run(self) -> None:
        self._initialize()
        try:
            self._install_sync_chunk()
            while True:
                self._tick()
        except KeyboardInterrupt:
            logger.info("Stopped by user.")
        finally:
            if self.camera_grabber is not None:
                self.camera_grabber.stop()
            for camera_cap in self.camera_caps.values():
                camera_cap.release()

    def _initialize(self) -> None:
        self.camera_caps = {
            name: VideoCapture(name)
            for name in get_camera_name(self.config.camera_config)
        }
        self.camera_grabber = CameraGrabber(self.camera_caps)
        self.camera_grabber.start()
        self.camera_grabber.wait_until_ready()

        self.main_client = server_client.PolicyClient(
            host=self.config.host,
            port=self.config.port,
            timeout_ms=self.config.timeout_ms,
            api_token=self.config.api_token,
        )
        print("Waiting for gr00t server to ping")
        if not self.main_client.ping():
            raise ConnectionError("Failed to connect to the gr00t server")
        self.main_client.reset()
        print("Server is alive!")

        initial_msg = wait_for_body_pose_msg(self.body_pose)
        initial_hand = wait_for_hand_state_msg(self.hand_subscriber)
        self.history.put(initial_msg, initial_hand)
        while True:
            self.root_pose = self.body_pose.get_root_pose()
            if self.root_pose is not None:
                print("root pose received!")
                break
            sleep(0.01)
        relative_reference_pose = self.root_pose.copy()
        if self.config.root_pose_z is not None:
            self.root_pose[2] = float(self.config.root_pose_z)
        pose15 = self.body_pose.get_15_pose7()
        if pose15 is None:
            raise RuntimeError("Failed to receive the initial 15-point body pose")
        root_tiled = np.repeat(
            relative_reference_pose[None, :], MOCAP_NUM_JOINTS, axis=0
        )
        pose15_rel = compute_relative(root_tiled, pose15)
        self.joint_rel_cum = pose15_rel[SELECT_11_INDICES, :3].astype(np.float32)

    def _tick(self) -> None:
        self._collect_async_result()
        self._activate_pending_if_ready()

        if self.active is None or self.active.done:
            if self.worker.busy:
                self._hold_last_frame()
                self.rate.sleep()
                return
            self._finish_active_chunk()
            self._install_sync_chunk()
            return

        self._maybe_submit_rtc()
        self._send_next_frame()
        self._sample_history()
        self._collect_async_result()
        self._activate_pending_if_ready()
        self.rate.sleep()

    def _install_sync_chunk(self) -> None:
        if self.main_client is None:
            raise RuntimeError("main_client is not initialized")
        result = self._infer_once(
            policy_client=self.main_client,
            chunk_start_root_rel=self.root_rel_cum,
            chunk_start_joint_rel=self.joint_rel_cum,
            rtc_overlap_steps=None,
            rtc_frozen_steps=None,
            submit_action_frame_count=None,
            parent_chunk_id=None,
        )
        self._install_chunk(result, action_frame_offset=0, allow_bridge=True)
        logger.info("installed sync action chunk")

    def _infer_once(
        self,
        policy_client,
        chunk_start_root_rel: np.ndarray,
        chunk_start_joint_rel: np.ndarray,
        rtc_overlap_steps: int | None,
        rtc_frozen_steps: int | None,
        submit_action_frame_count: int | None,
        parent_chunk_id: int | None,
    ) -> InferenceResult:
        state_history = self.history.get_all()
        if self.camera_grabber is None:
            raise RuntimeError("camera_grabber is not initialized")
        frame = self.camera_grabber.get_frames()
        observation = build_observation_from_msg_with_history(
            state_history,
            self.config.task_description,
            frame=frame,
        )

        if rtc_overlap_steps is not None:
            if rtc_frozen_steps is None or submit_action_frame_count is None:
                raise ValueError(
                    "RTC overlap, frozen steps, and submit frame "
                    "must be provided together"
                )
            observation.update(
                gr00t_rtc_inputs(
                    self.config,
                    self.action_horizon,
                    overlap_steps=rtc_overlap_steps,
                    frozen_steps=rtc_frozen_steps,
                )
            )

        response = policy_client.get_action(observation)
        measured_delay_steps = None
        if submit_action_frame_count is not None:
            elapsed_frames = max(
                self.sent_action_frame_count() - submit_action_frame_count,
                0,
            )
            measured_delay_steps = model_steps_from_frame_count(
                elapsed_frames,
                self.interp_stride,
            )
            logger.info(
                "GR00T RTC response delay: measured=%d, overlap=%s",
                measured_delay_steps,
                rtc_overlap_steps,
            )

        actions_rel = np.asarray(
            response[0]["mocap"][0], dtype=np.float32
        ).reshape(-1, 114)
        if actions_rel.shape[0] != self.action_horizon:
            raise ValueError(
                "Server action horizon changed during inference: "
                f"expected {self.action_horizon}, got {actions_rel.shape[0]}"
            )
        if self.root_pose is None:
            raise RuntimeError("root_pose is not initialized")
        actions_abs, hands_abs = model_action_to_abs_action(
            actions_rel,
            self.root_pose,
            chunk_start_root_rel,
            chunk_start_joint_rel,
        )
        return InferenceResult(
            actions_abs=actions_abs,
            hands_abs=hands_abs,
            actions_rel=actions_rel,
            chunk_start_root_rel=chunk_start_root_rel.copy(),
            chunk_start_joint_rel=chunk_start_joint_rel.copy(),
            rtc_overlap_steps=rtc_overlap_steps,
            rtc_frozen_steps=rtc_frozen_steps,
            submit_action_frame_count=submit_action_frame_count,
            parent_chunk_id=parent_chunk_id,
            measured_delay_steps=measured_delay_steps,
        )

    def _install_chunk(
        self,
        result: InferenceResult,
        action_frame_offset: int,
        allow_bridge: bool,
    ) -> None:
        full_frames, full_hands, prepended_bridge_frames = build_action_frames(
            prev_last_pose7=self.last_sent_pose,
            prev_last_hand=self.last_sent_hand,
            cur_chunk_pose7=result.actions_abs,
            cur_chunk_hand=result.hands_abs,
            enable_interp=self.config.enable_action_interp,
            allow_bridge=allow_bridge,
            num_interp=self.config.interp_num,
        )
        action_frame_count = full_frames.shape[0] - prepended_bridge_frames
        if action_frame_offset < 0 or action_frame_offset >= action_frame_count:
            raise ValueError(
                f"action_frame_offset={action_frame_offset} is outside action frames "
                f"with length={action_frame_count}"
            )

        if prepended_bridge_frames > 0 and action_frame_offset == 0:
            frames = full_frames
            hand_frames = full_hands
        else:
            frames = full_frames[prepended_bridge_frames + action_frame_offset:]
            hand_frames = full_hands[prepended_bridge_frames + action_frame_offset:]
            prepended_bridge_frames = 0

        self.active = ActiveChunk(
            chunk_id=self.next_chunk_id,
            result=result,
            frames=frames,
            hand_frames=hand_frames,
            action_frame_offset=action_frame_offset,
            prepended_bridge_frames=prepended_bridge_frames,
        )
        self.next_chunk_id += 1
        self.pending_rtc_result = None

    def _finish_active_chunk(self) -> None:
        if self.active is None:
            return
        self.root_rel_cum = self.active.current_root(self.interp_stride).copy()
        self.joint_rel_cum = self.active.joint_relative_at_step(
            self.active.executed_steps(self.interp_stride)
        ).copy()
        logger.info(
            "finished chunk id=%d at step=%d",
            self.active.chunk_id,
            self.active.executed_steps(self.interp_stride),
        )
        self.active = None

    def _send_next_frame(self) -> None:
        if self.active is None:
            raise RuntimeError("No active action chunk to send")

        mocap_frame, hand_frame = self.active.next_frame()
        xyz, wxyz = split_xyz_wxyz(mocap_frame)
        self.publisher.send_msg(xyz=xyz, wxyz=wxyz)
        self.hand_publisher.send(hand_frame)
        is_action_frame = self.active.mark_sent()
        self.last_sent_pose = mocap_frame.copy()
        self.last_sent_hand = hand_frame.copy()

        if is_action_frame:
            with self._counter_lock:
                self._sent_action_frame_count += 1

    def _hold_last_frame(self) -> None:
        if self.last_sent_pose is None or self.last_sent_hand is None:
            return
        xyz, wxyz = split_xyz_wxyz(self.last_sent_pose)
        self.publisher.send_msg(fps=self.config.send_fps, xyz=xyz, wxyz=wxyz)
        self.hand_publisher.send(self.last_sent_hand)

    def _maybe_submit_rtc(self) -> None:
        if (
            self.active is None
            or self.active.rtc_submitted
            or self.worker.busy
            or self.pending_rtc_result is not None
        ):
            return
        if not self.active.at_model_boundary(self.interp_stride):
            return

        action_exec_s = self.active.executed_steps(self.interp_stride)
        if action_exec_s <= 0 or action_exec_s >= self.action_horizon:
            return

        frozen_steps = self.delay_buffer.max()
        overlap_steps = self.action_horizon - action_exec_s
        remaining_frames = (
            self.active.action_frame_offset
            + self.active.frames.shape[0]
            - self.active.prepended_bridge_frames
            - self.active.full_frame_count()
        )
        if remaining_frames > self.config.rtc_submit_remaining_frames:
            return

        frozen_frames = frozen_steps * self.interp_stride
        if self.config.rtc_submit_remaining_frames <= frozen_frames:
            logger.warning(
                "GR00T RTC submit threshold is not larger than frozen delay: "
                "submit_remaining_frames=%d, frozen_frames=%d, frozen=%d",
                self.config.rtc_submit_remaining_frames,
                frozen_frames,
                frozen_steps,
            )
            return

        if frozen_steps >= overlap_steps:
            logger.warning(
                "GR00T RTC submit skipped: frozen=%d, overlap=%d, s=%d",
                frozen_steps,
                overlap_steps,
                action_exec_s,
            )
            return

        chunk_start_root_rel = self.active.root_at_step(action_exec_s)
        chunk_start_joint_rel = self.active.joint_relative_at_step(action_exec_s)
        submit_action_frame_count = self.sent_action_frame_count()
        ok = self.worker.submit(
            chunk_start_root_rel=chunk_start_root_rel,
            chunk_start_joint_rel=chunk_start_joint_rel,
            rtc_overlap_steps=overlap_steps,
            rtc_frozen_steps=frozen_steps,
            submit_action_frame_count=submit_action_frame_count,
            parent_chunk_id=self.active.chunk_id,
        )
        if ok:
            self.active.rtc_submitted = True
            logger.info(
                "submitted GR00T RTC: chunk=%d, s=%d, "
                "remaining_frames=%d, overlap=%d, frozen=%d",
                self.active.chunk_id,
                action_exec_s,
                remaining_frames,
                overlap_steps,
                frozen_steps,
            )

    def _collect_async_result(self) -> None:
        if not self.worker.poll_done():
            return
        try:
            result = self.worker.get_result()
        except Exception as exc:
            logger.error("async inference failed: %s", exc, exc_info=True)
            if not self.config.fallback_sync_on_async_error:
                raise
            return

        if self.active is None or result.parent_chunk_id != self.active.chunk_id:
            logger.warning(
                "discard async result for stale chunk: result_parent=%s, active=%s",
                result.parent_chunk_id,
                None if self.active is None else self.active.chunk_id,
            )
            if result.measured_delay_steps is not None:
                self.delay_buffer.add(result.measured_delay_steps)
            return
        self.pending_rtc_result = result
        logger.info("async inference result is pending activation")

    def _activate_pending_if_ready(self) -> bool:
        if self.pending_rtc_result is None or self.active is None:
            return False
        result = self.pending_rtc_result
        if result.submit_action_frame_count is None:
            raise ValueError(
                "Pending RTC result is missing submit_action_frame_count"
            )
        if result.rtc_overlap_steps is None or result.rtc_frozen_steps is None:
            raise ValueError("Pending RTC result is missing overlap or frozen steps")
        if result.parent_chunk_id != self.active.chunk_id:
            logger.warning(
                "discard pending result for stale chunk: result_parent=%s, active=%d",
                result.parent_chunk_id,
                self.active.chunk_id,
            )
            self.pending_rtc_result = None
            return False

        elapsed_frames = (
            self.sent_action_frame_count() - result.submit_action_frame_count
        )
        if elapsed_frames < 0:
            raise RuntimeError(
                "Action frame counter moved backwards while activating RTC result"
            )

        max_valid_frames = (
            min(result.rtc_overlap_steps, result.rtc_frozen_steps)
            * self.interp_stride
        )
        if elapsed_frames > max_valid_frames:
            stale_delay_steps = model_steps_from_frame_count(
                elapsed_frames,
                self.interp_stride,
            )
            self.delay_buffer.add(stale_delay_steps)
            logger.error(
                "discard stale GR00T RTC result: elapsed_frames=%d, "
                "valid_frames=%d, delay_steps=%d, buffer_max=%d",
                elapsed_frames,
                max_valid_frames,
                stale_delay_steps,
                self.delay_buffer.max(),
            )
            self.pending_rtc_result = None
            return False

        if elapsed_frames % self.interp_stride != 0:
            return False

        skipped_steps = elapsed_frames // self.interp_stride
        assert skipped_steps <= result.rtc_overlap_steps
        assert skipped_steps <= result.rtc_frozen_steps

        self.root_rel_cum = self.active.current_root(self.interp_stride).copy()
        self.joint_rel_cum = self.active.joint_relative_at_step(
            self.active.executed_steps(self.interp_stride)
        ).copy()
        action_frame_offset = action_frame_offset_after_steps(
            skipped_steps,
            self.interp_stride,
        )
        self.delay_buffer.add(skipped_steps)
        self._install_chunk(
            result,
            action_frame_offset=action_frame_offset,
            allow_bridge=False,
        )
        logger.info(
            "activated GR00T RTC result: skipped=%d, "
            "action_frame_offset=%d, buffer_max=%d",
            skipped_steps,
            action_frame_offset,
            self.delay_buffer.max(),
        )
        return True

    def _sample_history(self) -> None:
        if self.active is None or not self.active.at_model_boundary(
            self.interp_stride
        ):
            return
        msg = self.body_pose.get_msg()
        hand_state = self.hand_subscriber.get_state()
        if msg is not None and hand_state is not None:
            self.history.put(msg, hand_state)

    def sent_action_frame_count(self) -> int:
        with self._counter_lock:
            return self._sent_action_frame_count


def main() -> None:
    config = tyro.cli(ClientConfig)
    validate_config(config)
    runner = Gr00tRtcV2Runner(config)
    runner.run()


if __name__ == "__main__":
    main()

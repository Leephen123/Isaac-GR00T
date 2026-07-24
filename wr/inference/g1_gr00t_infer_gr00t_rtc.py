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

root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))
sys.path.append(os.getcwd())

from data_res.camera import VideoCapture
from data_res.dds import (
    MOCAP_NUM_JOINTS,
    MOCAP_POS_DIM,
    MOCAP_QUAT_DIM,
    BodyPoseConfig,
    BodyPoseSubscriber,
    MocapConfig,
    MocapUE5G115MsgPublisher,
    WR_GAE_BodyPose_Msg,
)
from data_res.log import get_logger
from data_res.transforms import (
    compute_absolute,
    interpolate_pose7,
    normalize_quaternion,
    restore_mocap_from_root_relative,
    rotation_6d_to_quaternion,
)
from data_res.utils import CAMERAS_MAP, SELECT_11_INDICES, get_camera_name
from inference.utils import msg_to_6d_rot_joint
from serve_res.gr00t import server_client

logger = get_logger(__name__)


@dataclass
class ClientConfig:
    host: str = "192.168.123.165"
    port: int = 9002
    timeout_ms: int = 15000
    api_token: str | None = None
    task_description: str = (
        "Clean the items on the desk and put the doll on the left sofa"
    )
    send_fps: float = 70.0
    history_len: int = 50
    action_horizon: int = 50
    state_sample_every: int = 3
    camera_flush_infer: int = 5
    rtc_init_delay_steps: int = 15
    rtc_delay_buffer_len: int = 3
    rtc_ramp_rate: float = 1.0
    fallback_sync_on_async_error: bool = True
    interp_num: int = 2
    enable_chunk_bridge_interp: bool = True
    chunk_bridge_interp_num: int = 5
    async_start_ratio: float = 0.98
    root_pose_z: float | None = 1.0
    fall_max_tilt_deg: float = 45.0
    fall_confirm_samples: int = 2
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
    actions_rel: np.ndarray
    chunk_start_root_rel: np.ndarray
    next_root_rel: np.ndarray
    rtc_overlap_steps: int | None
    rtc_frozen_steps: int | None
    inference_start_exec_frame: int | None
    executed_frame_offset: int | None
    measured_delay_steps: int | None


class DelayBuffer:
    def __init__(self, initial_delay_steps: int, maxlen: int = 3):
        if maxlen <= 0:
            raise ValueError(f"maxlen must be positive, got {maxlen}")
        self._lock = threading.Lock()
        self._queue = deque([max(int(initial_delay_steps), 0)], maxlen=maxlen)

    def add(self, delay_steps: int) -> None:
        with self._lock:
            self._queue.append(max(int(delay_steps), 0))

    def max(self) -> int:
        with self._lock:
            return max(self._queue)


class RobotSafetyMonitor:
    def __init__(self, max_tilt_deg: float, confirm_samples: int):
        self._max_tilt_deg = float(max_tilt_deg)
        self._confirm_samples = int(confirm_samples)
        self._lock = threading.Lock()
        self._tilted_samples = 0
        self._error: RuntimeError | None = None

    def observe(self, msg: WR_GAE_BodyPose_Msg) -> None:
        raw_quat = np.asarray(msg.wxyz, dtype=np.float32).reshape(1, 4)
        quat_norm = float(np.linalg.norm(raw_quat[0]))
        if not np.isfinite(raw_quat).all() or quat_norm < 0.5:
            with self._lock:
                if self._error is None:
                    self._error = RuntimeError(
                        f"Invalid body root quaternion with norm={quat_norm}"
                    )
            self.raise_if_unsafe()
        quat = normalize_quaternion(raw_quat)[0]
        _, qx, qy, _ = quat
        up_z = float(np.clip(1.0 - 2.0 * (qx * qx + qy * qy), -1.0, 1.0))
        tilt_deg = float(np.degrees(np.arccos(up_z)))

        with self._lock:
            if tilt_deg > self._max_tilt_deg:
                self._tilted_samples += 1
            else:
                self._tilted_samples = 0

            if (
                self._error is None
                and self._tilted_samples >= self._confirm_samples
            ):
                self._error = RuntimeError(
                    "Robot fall detected: "
                    f"root tilt={tilt_deg:.1f} deg, "
                    f"limit={self._max_tilt_deg:.1f} deg"
                )

        self.raise_if_unsafe()

    def raise_if_unsafe(self) -> None:
        with self._lock:
            error = self._error
        if error is not None:
            raise error


def split_xyz_wxyz(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    expected_shape = (MOCAP_NUM_JOINTS, MOCAP_POS_DIM + MOCAP_QUAT_DIM)
    frame = np.asarray(frame, dtype=np.float32)
    if frame.shape != expected_shape:
        raise ValueError(
            f"Expected mocap frame shape {expected_shape}, got {frame.shape}"
        )
    xyz = frame[:, :MOCAP_POS_DIM]
    wxyz = normalize_quaternion(frame[:, MOCAP_POS_DIM:])
    quat_norms = np.linalg.norm(wxyz, axis=-1)
    if np.any(quat_norms < 0.5):
        raise ValueError("Refusing to send mocap action with invalid quaternion")
    return xyz, wxyz


def bridge_action_chunks(
    prev_last_pose7: np.ndarray | None,
    cur_chunk_pose7: np.ndarray,
    enable: bool = True,
    num_interp: int = 3,
) -> np.ndarray:
    cur_chunk_pose7 = np.asarray(cur_chunk_pose7, dtype=np.float32)

    if not enable:
        return cur_chunk_pose7

    if prev_last_pose7 is None:
        return cur_chunk_pose7

    if num_interp <= 0:
        return cur_chunk_pose7

    if cur_chunk_pose7.ndim != 3 or cur_chunk_pose7.shape[1:] != (15, 7):
        raise ValueError(
            f"Expected cur_chunk_pose7 shape (T,15,7), got {cur_chunk_pose7.shape}"
        )

    prev_last_pose7 = np.asarray(prev_last_pose7, dtype=np.float32)

    if prev_last_pose7.shape != (15, 7):
        raise ValueError(
            f"Expected prev_last_pose7 shape (15,7), got {prev_last_pose7.shape}"
        )

    if cur_chunk_pose7.shape[0] == 0:
        return cur_chunk_pose7

    bridge_pair = np.stack(
        [
            prev_last_pose7,
            cur_chunk_pose7[0],
        ],
        axis=0,
    )

    bridge = interpolate_pose7(bridge_pair, num_interp=num_interp)
    inserted = bridge[1:-1]

    if inserted.shape[0] == 0:
        return cur_chunk_pose7

    out = np.concatenate([inserted, cur_chunk_pose7], axis=0).astype(np.float32)

    print(
        f"[CHUNK BRIDGE] enabled | "
        f"inserted={inserted.shape[0]} | "
        f"before={cur_chunk_pose7.shape[0]} | "
        f"after={out.shape[0]}"
    )

    return out


def load_image_from_path_or_array(
    image_input: Path | np.ndarray,
    target_size: tuple[int, int] = (256, 256),
    debug: bool = False,
    name: str | None = None,
) -> np.ndarray:
    if isinstance(image_input, Path):
        image = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, np.ndarray):
        image = np.asarray(image_input)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Expected frame shape (H, W, 3), got {image.shape}")
        image = Image.fromarray(image.astype(np.uint8)[:, :, ::-1]).convert("RGB")
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    if debug:
        os.makedirs("./image_debug", exist_ok=True)
        save_name = name if name is not None else "debug_image"
        image.save(os.path.join("./image_debug", f"{save_name}.png"))

    return np.asarray(image.resize(target_size, Image.BILINEAR))[None, None, ...]


def build_observation_from_msg_with_history(
    history_state: np.ndarray,
    task_description: str,
    frame: dict[str, np.ndarray] | None = None,
) -> dict:
    history_state = np.asarray(history_state, dtype=np.float32)
    video = {
        "ego_view": np.zeros((1, 1, 256, 256, 3), dtype=np.uint8),
        "left_wrist_view": np.zeros((1, 1, 256, 256, 3), dtype=np.uint8),
        "right_wrist_view": np.zeros((1, 1, 256, 256, 3), dtype=np.uint8),
    }
    if frame is not None:
        for name, image in frame.items():
            video[CAMERAS_MAP[name]] = load_image_from_path_or_array(
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


class StateHistoryQueue:
    def __init__(self, maxlen: int = 50):
        self._lock = threading.Lock()
        self._queue = deque(maxlen=maxlen)

    def put(self, msg: WR_GAE_BodyPose_Msg) -> None:
        with self._lock:
            self._queue.append(msg_to_6d_rot_joint(msg))

    def put_and_get_all(self, msg: WR_GAE_BodyPose_Msg) -> np.ndarray:
        with self._lock:
            self._queue.append(msg_to_6d_rot_joint(msg))
            return self._copy_all_locked()

    def get_all(self, poll_interval_s: float = 0.01) -> np.ndarray:
        has_warned = False
        while True:
            with self._lock:
                if self._queue:
                    return self._copy_all_locked()

            if not has_warned:
                logger.warning("History queue is empty; waiting for data.")
                has_warned = True
            sleep(poll_interval_s)

    def _copy_all_locked(self) -> np.ndarray:
        states = [state.copy() for state in self._queue]
        if len(states) < self._queue.maxlen:
            states = [states[0].copy()] * (
                self._queue.maxlen - len(states)
            ) + states
        return np.stack(states, axis=0)


class MocapDataQueue:
    def __init__(self, maxlen: int = 300):
        self._lock = threading.Lock()
        self._queue = deque(maxlen=maxlen)
        self._last_xyz: np.ndarray | None = None
        self._last_wxyz: np.ndarray | None = None
        self._executed_count = 0
        self._in_flight = False
        self._sender_error: BaseException | None = None

    def put(self, xyz: np.ndarray, wxyz: np.ndarray) -> None:
        xyz = np.asarray(xyz, dtype=np.float32).reshape(
            MOCAP_NUM_JOINTS, MOCAP_POS_DIM
        ).copy()
        wxyz = np.asarray(wxyz, dtype=np.float32).reshape(
            MOCAP_NUM_JOINTS, MOCAP_QUAT_DIM
        ).copy()
        if not np.isfinite(xyz).all() or not np.isfinite(wxyz).all():
            raise ValueError("Refusing to queue non-finite mocap action")
        with self._lock:
            if self._sender_error is not None:
                raise RuntimeError("Mocap sender thread failed") from self._sender_error
            self._queue.append((xyz, wxyz))
            self._last_xyz = xyz
            self._last_wxyz = wxyz

    def get_next_or_last(
        self,
        should_stop: Callable[[], bool] | None = None,
        poll_interval_s: float = 0.01,
    ) -> tuple[np.ndarray, np.ndarray, bool] | None:
        while True:
            with self._lock:
                if len(self._queue) > 0:
                    xyz, wxyz = self._queue.popleft()
                    self._last_xyz = xyz
                    self._last_wxyz = wxyz
                    self._in_flight = True
                    return xyz, wxyz, True
                if self._last_xyz is not None and self._last_wxyz is not None:
                    return self._last_xyz, self._last_wxyz, False

            if should_stop is not None and should_stop():
                return None
            sleep(poll_interval_s)

    def mark_executed(self) -> None:
        with self._lock:
            self._executed_count += 1
            self._in_flight = False

    def mark_sender_failed(self, error: BaseException) -> None:
        with self._lock:
            self._sender_error = error
            self._in_flight = False

    def raise_if_sender_failed(self) -> None:
        with self._lock:
            error = self._sender_error
        if error is not None:
            raise RuntimeError("Mocap sender thread failed") from error

    def clear(self, reset_last: bool = False) -> None:
        with self._lock:
            self._queue.clear()
            if reset_last:
                self._last_xyz = None
                self._last_wxyz = None

    def size(self) -> int:
        with self._lock:
            return len(self._queue)

    def empty(self) -> bool:
        with self._lock:
            return len(self._queue) == 0 and not self._in_flight

    def is_full(self) -> bool:
        with self._lock:
            return len(self._queue) >= self._queue.maxlen

    def executed_count(self) -> int:
        with self._lock:
            return self._executed_count


class MocapSenderThread(threading.Thread):
    def __init__(
        self,
        mocap_queue: MocapDataQueue,
        fps: float,
        mocap_cfg: MocapConfig,
    ):
        super().__init__(daemon=True)
        self.mocap_queue = mocap_queue
        self.fps = fps
        self.publisher = MocapUE5G115MsgPublisher(mocap_cfg)
        self.running = True
        self.rate = RateLimiter(frequency=fps)

    def run(self) -> None:
        try:
            while self.running:
                item = self.mocap_queue.get_next_or_last(
                    should_stop=lambda: not self.running
                )
                if item is None:
                    break
                xyz, wxyz, is_new_action = item
                self.publisher.send_msg(fps=self.fps, xyz=xyz, wxyz=wxyz)
                if is_new_action:
                    self.mocap_queue.mark_executed()
                self.rate.sleep()
        except BaseException as exc:
            self.mocap_queue.mark_sender_failed(exc)
            self.running = False
            logger.exception("Mocap sender failed; stopping inference.")

    def stop(self) -> None:
        self.running = False


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

    def _make_async_client(self):
        return server_client.PolicyClient(
            host=self._config.host,
            port=self._config.port,
            timeout_ms=self._config.timeout_ms,
            api_token=self._config.api_token,
        )

    def _run(self, **kwargs: Any) -> None:
        try:
            async_client = self._make_async_client()
            result = self._infer_fn(policy_client=async_client, **kwargs)
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
            self._result = None
            self._error = None

        if error is not None:
            raise error
        if result is None:
            raise RuntimeError("inference finished without result")
        return result


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
    body_pose: BodyPoseSubscriber,
    poll_interval_s: float = 0.01,
) -> WR_GAE_BodyPose_Msg:
    while True:
        msg = body_pose.get_msg()
        if msg is not None:
            return msg
        sleep(poll_interval_s)


def _waiting_robot_move_done(mocap_queue: MocapDataQueue) -> bool:
    while True:
        mocap_queue.raise_if_sender_failed()
        if not mocap_queue.empty():
            sleep(0.01)
            continue
        return True


def model_action_to_abs_action(
    action_output: np.ndarray,
    init_pose: np.ndarray,
    chunk_start_root_rel: np.ndarray,
    interp_num: int,
) -> tuple[np.ndarray, np.ndarray]:
    action_output = np.asarray(action_output, dtype=np.float32).reshape(-1, 102)
    if not np.isfinite(action_output).all():
        raise ValueError("Model returned non-finite actions")
    if not np.any(np.abs(action_output) > 1e-6):
        raise ValueError("Model returned an all-zero action chunk")
    root_delta = action_output[:, :3]
    action_without_root_delta = action_output[:, 3:]
    action_mocap_xyz = action_without_root_delta[:, :33].reshape(-1, 11, 3)
    action_mocap_6d = action_without_root_delta[:, 33:].reshape(-1, 11, 6)
    action_mocap = np.concatenate(
        [action_mocap_xyz, action_mocap_6d],
        axis=-1,
    ).reshape(-1, 99)
    action_mocap = np.concatenate([root_delta, action_mocap], axis=-1)
    action_11x9, next_root_rel = restore_mocap_from_root_relative(
        action_mocap,
        chunk_start_root_rel,
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
    action_abs = interpolate_pose7(action_abs, interp_num)
    if not np.isfinite(action_abs).all() or not np.isfinite(next_root_rel).all():
        raise ValueError("Restored mocap action contains non-finite values")
    return action_abs, next_root_rel


def validate_config(config: ClientConfig) -> None:
    if config.send_fps <= 0:
        raise ValueError(f"send_fps must be positive, got {config.send_fps}")
    if config.history_len <= 0:
        raise ValueError(f"history_len must be positive, got {config.history_len}")
    if config.state_sample_every <= 0:
        raise ValueError(
            f"state_sample_every must be positive, got {config.state_sample_every}"
        )
    if config.camera_flush_infer <= 0:
        raise ValueError(
            f"camera_flush_infer must be positive, got {config.camera_flush_infer}"
        )
    if config.interp_num < 0:
        raise ValueError(f"interp_num must be non-negative, got {config.interp_num}")
    if config.chunk_bridge_interp_num < 0:
        raise ValueError(
            "chunk_bridge_interp_num must be non-negative, "
            f"got {config.chunk_bridge_interp_num}"
        )
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
    if not 0 <= config.async_start_ratio <= 1:
        raise ValueError(
            f"async_start_ratio must be in [0, 1], got {config.async_start_ratio}"
        )
    if not 0 < config.fall_max_tilt_deg < 180:
        raise ValueError(
            "fall_max_tilt_deg must be in (0, 180), "
            f"got {config.fall_max_tilt_deg}"
        )
    if config.fall_confirm_samples <= 0:
        raise ValueError(
            "fall_confirm_samples must be positive, "
            f"got {config.fall_confirm_samples}"
        )


# def get_action_horizon(policy_client) -> int:
#     modality_config = policy_client.get_modality_config()
#     action_config = modality_config.get("action")
#     delta_indices = (
#         action_config.get("delta_indices")
#         if isinstance(action_config, dict)
#         else getattr(action_config, "delta_indices", None)
#     )
#     if delta_indices is None or len(delta_indices) == 0:
#         raise ValueError("Server action modality config is missing delta_indices")
#     return len(delta_indices)


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


def chunk_root_rel_at_step(
    result: InferenceResult,
    action_exec_s: int,
) -> np.ndarray:
    root_delta = result.actions_rel[:action_exec_s, :3].sum(axis=0)
    return result.chunk_start_root_rel + root_delta


def main() -> None:
    config = tyro.cli(ClientConfig)
    validate_config(config)

    body_pose = BodyPoseSubscriber(config.body_pose_cfg)
    mocap_queue = MocapDataQueue(maxlen=300)
    state_history = StateHistoryQueue(maxlen=config.history_len)

    camera_name_list = get_camera_name(config.camera_config)
    camera_caps = {name: VideoCapture(name) for name in camera_name_list}
    sleep(2)

    main_client = server_client.PolicyClient(
        host=config.host,
        port=config.port,
        timeout_ms=config.timeout_ms,
        api_token=config.api_token,
    )
    print("Waiting for gr00t server to ping")
    if not main_client.ping():
        print("Failed to connect to the server.")
        sys.exit(1)
    main_client.reset()
    print("Server is alive!")
    action_horizon = config.action_horizon

    initial_body_pose_msg = wait_for_body_pose_msg(body_pose)
    root_pose = body_pose.msg_to_root_pose(initial_body_pose_msg)
    if config.root_pose_z is not None:
        root_pose[2] = float(config.root_pose_z)

    sender_thread = MocapSenderThread(
        mocap_queue=mocap_queue,
        fps=config.send_fps,
        mocap_cfg=config.mocap_cfg,
    )
    sender_thread.start()
    logger.info("Sender thread started.")

    interp_stride = config.interp_num + 1
    delay_buffer = DelayBuffer(
        initial_delay_steps=config.rtc_init_delay_steps,
        maxlen=config.rtc_delay_buffer_len,
    )
    safety_monitor = RobotSafetyMonitor(
        max_tilt_deg=config.fall_max_tilt_deg,
        confirm_samples=config.fall_confirm_samples,
    )
    safety_monitor.observe(initial_body_pose_msg)

    def infer_once(
        policy_client,
        msg: WR_GAE_BodyPose_Msg,
        frame: dict[str, np.ndarray],
        chunk_start_root_rel: np.ndarray,
        rtc_overlap_steps: int | None,
        rtc_frozen_steps: int | None,
        inference_start_exec_frame: int | None,
    ) -> InferenceResult:
        safety_monitor.observe(msg)
        state_history_snapshot = state_history.put_and_get_all(msg)
        observation = build_observation_from_msg_with_history(
            state_history_snapshot,
            config.task_description,
            frame=frame,
        )
        if rtc_overlap_steps is not None:
            if rtc_frozen_steps is None or inference_start_exec_frame is None:
                raise ValueError(
                    "RTC overlap, frozen steps, and inference start frame "
                    "must be provided together"
                )
            rtc_inputs = gr00t_rtc_inputs(
                config,
                action_horizon,
                overlap_steps=rtc_overlap_steps,
                frozen_steps=rtc_frozen_steps,
            )
            observation.update(rtc_inputs)

        response = policy_client.get_action(observation)
        measured_delay_steps = None
        if inference_start_exec_frame is not None:
            elapsed_exec_frames = max(
                mocap_queue.executed_count() - inference_start_exec_frame,
                0,
            )
            measured_delay_steps = (
                elapsed_exec_frames + interp_stride - 1
            ) // interp_stride
            logger.info(
                "GR00T RTC response delay: measured=%d, overlap=%d",
                measured_delay_steps,
                rtc_overlap_steps,
            )

        actions_rel = np.asarray(
            response[0]["mocap"][0], dtype=np.float32
        ).reshape(-1, 102)
        if actions_rel.shape[0] != action_horizon:
            raise ValueError(
                "Server action horizon changed during inference: "
                f"expected {action_horizon}, got {actions_rel.shape[0]}"
            )
        actions_abs, next_root_rel = model_action_to_abs_action(
            actions_rel,
            root_pose,
            chunk_start_root_rel,
            config.interp_num,
        )
        return InferenceResult(
            actions_abs=actions_abs,
            actions_rel=actions_rel,
            chunk_start_root_rel=chunk_start_root_rel.copy(),
            next_root_rel=next_root_rel.copy(),
            rtc_overlap_steps=rtc_overlap_steps,
            rtc_frozen_steps=rtc_frozen_steps,
            inference_start_exec_frame=inference_start_exec_frame,
            executed_frame_offset=None,
            measured_delay_steps=measured_delay_steps,
        )

    def model_steps_from_frame_count(frame_count: int) -> int:
        if frame_count <= 0:
            return 0
        return (frame_count + interp_stride - 1) // interp_stride

    def prepare_result_for_execution(result: InferenceResult) -> InferenceResult:
        if result.executed_frame_offset is not None:
            return result

        if result.inference_start_exec_frame is None:
            result.executed_frame_offset = 0
            return result

        if result.rtc_overlap_steps is None or result.rtc_frozen_steps is None:
            raise ValueError("RTC result is missing overlap or frozen steps")

        elapsed_exec_frames = max(
            mocap_queue.executed_count() - result.inference_start_exec_frame,
            0,
        )
        if elapsed_exec_frames % interp_stride != 0:
            raise RuntimeError(
                "GR00T RTC result must switch at a model-step boundary, "
                f"got elapsed_frames={elapsed_exec_frames}, stride={interp_stride}"
            )

        skipped_steps = elapsed_exec_frames // interp_stride
        delay_buffer.add(skipped_steps)
        if skipped_steps > result.rtc_overlap_steps:
            raise RuntimeError(
                "GR00T RTC result is stale: "
                f"skipped={skipped_steps}, overlap={result.rtc_overlap_steps}"
            )
        if skipped_steps > result.rtc_frozen_steps:
            raise RuntimeError(
                "GR00T RTC result exceeded its frozen prefix: "
                f"skipped={skipped_steps}, frozen={result.rtc_frozen_steps}"
            )

        # For skipped_steps > 0, the old chunk has already reached the same
        # model action as this result's skipped prefix. Keep the interpolation
        # frames after that action so execution continues smoothly.
        executed_frame_offset = (
            0
            if skipped_steps == 0
            else (skipped_steps - 1) * interp_stride + 1
        )
        result.actions_abs = result.actions_abs[executed_frame_offset:]
        result.executed_frame_offset = executed_frame_offset
        logger.info(
            "activate GR00T RTC result: skipped=%d, overlap=%d, "
            "frame_offset=%d, buffer_max=%d",
            skipped_steps,
            result.rtc_overlap_steps,
            executed_frame_offset,
            delay_buffer.max(),
        )
        return result

    inference_worker = AsyncInferenceWorker(infer_once, config)
    current_result: InferenceResult | None = None
    prefetched_result: InferenceResult | None = None
    global_state_sample_every = 0
    last_chunk_last_pose7: np.ndarray | None = None
    root_rel_cum = np.zeros(3, dtype=np.float32)

    try:
        while True:
            safety_monitor.raise_if_unsafe()
            _waiting_robot_move_done(mocap_queue)

            if prefetched_result is not None:
                try:
                    current_result = prepare_result_for_execution(prefetched_result)
                    logger.info("use prefetched async action chunk")
                except Exception as exc:
                    safety_monitor.raise_if_unsafe()
                    mocap_queue.raise_if_sender_failed()
                    logger.error(
                        "prefetched GR00T RTC result cannot be activated: %s",
                        exc,
                        exc_info=True,
                    )
                    if not config.fallback_sync_on_async_error:
                        raise
                    current_result = None
                prefetched_result = None
            else:
                if inference_worker.busy:
                    logger.info("waiting async inference result because queue is empty")
                    while inference_worker.busy:
                        latest_msg = body_pose.get_msg()
                        if latest_msg is not None:
                            safety_monitor.observe(latest_msg)
                        mocap_queue.raise_if_sender_failed()
                        sleep(0.01)

                if inference_worker.poll_done():
                    try:
                        current_result = prepare_result_for_execution(
                            inference_worker.get_result()
                        )
                        logger.info("use late async action chunk")
                    except Exception as exc:
                        safety_monitor.raise_if_unsafe()
                        mocap_queue.raise_if_sender_failed()
                        logger.error("async inference failed: %s", exc, exc_info=True)
                        if not config.fallback_sync_on_async_error:
                            raise
                        current_result = None
                else:
                    current_result = None

                if current_result is None:
                    while True:
                        msg = body_pose.get_msg()
                        if msg is not None:
                            print("Msg received!")
                            break
                        sleep(0.01)

                    frame = read_camera_frames(
                        camera_caps, flush_count=config.camera_flush_infer
                    )
                    current_result = infer_once(
                        policy_client=main_client,
                        msg=msg,
                        frame=frame,
                        chunk_start_root_rel=root_rel_cum,
                        rtc_overlap_steps=None,
                        rtc_frozen_steps=None,
                        inference_start_exec_frame=None,
                    )
                    current_result = prepare_result_for_execution(current_result)
                    logger.info("use sync action chunk")

            if current_result is None:
                msg = wait_for_body_pose_msg(body_pose)
                frame = read_camera_frames(
                    camera_caps, flush_count=config.camera_flush_infer
                )
                current_result = infer_once(
                    policy_client=main_client,
                    msg=msg,
                    frame=frame,
                    chunk_start_root_rel=root_rel_cum,
                    rtc_overlap_steps=None,
                    rtc_frozen_steps=None,
                    inference_start_exec_frame=None,
                )
                current_result = prepare_result_for_execution(current_result)
                logger.info("use sync action chunk after rejected async result")

            if current_result.executed_frame_offset is None:
                raise RuntimeError("action result was not prepared for execution")

            enable_chunk_bridge = (
                config.enable_chunk_bridge_interp
                and current_result.rtc_overlap_steps is None
            )
            action_concat_abs = bridge_action_chunks(
                prev_last_pose7=last_chunk_last_pose7,
                cur_chunk_pose7=current_result.actions_abs,
                enable=enable_chunk_bridge,
                num_interp=config.chunk_bridge_interp_num,
            )
            bridge_frame_count = (
                action_concat_abs.shape[0] - current_result.actions_abs.shape[0]
            )
            async_submitted_for_this_chunk = False
            switched_to_prefetched = False
            ratio_async_start_t = max(
                0,
                min(
                    action_concat_abs.shape[0] - 1,
                    int(action_concat_abs.shape[0] * config.async_start_ratio),
                )
            )
            frozen_steps = delay_buffer.max()
            executed_steps_before_chunk = model_steps_from_frame_count(
                current_result.executed_frame_offset
            )
            # Keep at least one model step after the frozen prefix so the
            # server-side GR00T RTC ramp is not collapsed to an empty interval.
            max_submit_s = action_horizon - frozen_steps - 1
            if max_submit_s <= executed_steps_before_chunk:
                async_start_t = action_concat_abs.shape[0]
                logger.warning(
                    "GR00T RTC cannot be submitted from this chunk: "
                    "executed_offset=%d, frozen=%d, H=%d",
                    executed_steps_before_chunk,
                    frozen_steps,
                    action_horizon,
                )
            else:
                ratio_sent_action_frames = max(
                    ratio_async_start_t + 1 - bridge_frame_count,
                    1,
                )
                ratio_full_frame_count = (
                    current_result.executed_frame_offset
                    + ratio_sent_action_frames
                )
                ratio_submit_s = (
                    (ratio_full_frame_count - 1 + interp_stride - 1)
                    // interp_stride
                    + 1
                )
                target_submit_s = min(
                    max(ratio_submit_s, executed_steps_before_chunk + 1),
                    max_submit_s,
                )
                target_full_frame_count = (
                    1 + (target_submit_s - 1) * interp_stride
                )
                target_sent_action_frames = (
                    target_full_frame_count
                    - current_result.executed_frame_offset
                )
                async_start_t = (
                    bridge_frame_count + target_sent_action_frames - 1
                )

            for t in range(action_concat_abs.shape[0]):
                safety_monitor.raise_if_unsafe()
                mocap_frame = action_concat_abs[t]
                xyz, wxyz = split_xyz_wxyz(mocap_frame)
                mocap_queue.put(xyz, wxyz)
                logger.debug(f"put mocap frame {t} to queue")

                global_state_sample_every += 1

                if (not async_submitted_for_this_chunk) and t >= async_start_t:
                    try:
                        # waiting for the previous actions to finish
                        _waiting_robot_move_done(mocap_queue)
                        async_msg = wait_for_body_pose_msg(body_pose)
                        safety_monitor.observe(async_msg)
                        async_frame = read_camera_frames(
                            camera_caps,
                            flush_count=config.camera_flush_infer,
                        )
                        sent_action_frames = max(
                            0,
                            t + 1 - bridge_frame_count,
                        )
                        full_frame_count = (
                            current_result.executed_frame_offset
                            + sent_action_frames
                        )
                        if (full_frame_count - 1) % interp_stride != 0:
                            raise RuntimeError(
                                "GR00T RTC inference must be submitted at a "
                                "model-step boundary, "
                                f"got full_frame_count={full_frame_count}, "
                                f"stride={interp_stride}"
                            )
                        action_exec_s = model_steps_from_frame_count(
                            full_frame_count
                        )
                        if action_exec_s > 0:
                            real_overlap_steps = action_horizon - action_exec_s
                            frozen_steps = delay_buffer.max()
                            if frozen_steps >= real_overlap_steps:
                                raise RuntimeError(
                                    "GR00T RTC inference was submitted too late: "
                                    f"frozen={frozen_steps}, "
                                    f"real_overlap={real_overlap_steps}, "
                                    f"action_exec={action_exec_s}, "
                                    f"H={action_horizon}"
                                )
                            next_chunk_start_root_rel = chunk_root_rel_at_step(
                                current_result,
                                action_exec_s,
                            )
                            inference_start_exec_frame = (
                                mocap_queue.executed_count()
                            )
                            ok = inference_worker.submit(
                                msg=async_msg,
                                frame=async_frame,
                                chunk_start_root_rel=next_chunk_start_root_rel,
                                rtc_overlap_steps=real_overlap_steps,
                                rtc_frozen_steps=frozen_steps,
                                inference_start_exec_frame=inference_start_exec_frame,
                            )
                            if ok:
                                async_submitted_for_this_chunk = True
                                logger.info(
                                    "async inference submitted at t=%d/%d, "
                                    "queue=%d, overlap=%d, frozen=%d",
                                    t,
                                    action_concat_abs.shape[0],
                                    mocap_queue.size(),
                                    real_overlap_steps,
                                    frozen_steps,
                                )
                    except Exception as exc:
                        safety_monitor.raise_if_unsafe()
                        mocap_queue.raise_if_sender_failed()
                        logger.warning("submit async inference failed: %s", exc)
                        async_submitted_for_this_chunk = True

                while True:
                    if global_state_sample_every >= config.state_sample_every:
                        _waiting_robot_move_done(mocap_queue)
                        hist_msg = body_pose.get_msg()
                        if hist_msg is not None:
                            safety_monitor.observe(hist_msg)
                            state_history.put(hist_msg)
                        global_state_sample_every = 0
                    break

                if inference_worker.poll_done() and prefetched_result is None:
                    try:
                        prefetched_result = inference_worker.get_result()
                        logger.info("async inference finished and cached")
                    except Exception as exc:
                        safety_monitor.raise_if_unsafe()
                        mocap_queue.raise_if_sender_failed()
                        logger.error("async inference failed: %s", exc, exc_info=True)
                        prefetched_result = None

                if (
                    prefetched_result is not None
                    and prefetched_result.inference_start_exec_frame is not None
                    and prefetched_result.rtc_overlap_steps is not None
                    and prefetched_result.rtc_frozen_steps is not None
                ):
                    _waiting_robot_move_done(mocap_queue)
                    elapsed_exec_frames = max(
                        mocap_queue.executed_count()
                        - prefetched_result.inference_start_exec_frame,
                        0,
                    )
                    max_valid_frames = (
                        min(
                            prefetched_result.rtc_overlap_steps,
                            prefetched_result.rtc_frozen_steps,
                        )
                        * interp_stride
                    )
                    if elapsed_exec_frames > max_valid_frames:
                        stale_delay_steps = model_steps_from_frame_count(
                            elapsed_exec_frames
                        )
                        delay_buffer.add(stale_delay_steps)
                        logger.error(
                            "discard stale GR00T RTC result: elapsed_frames=%d, "
                            "valid_frames=%d, delay_steps=%d, buffer_max=%d",
                            elapsed_exec_frames,
                            max_valid_frames,
                            stale_delay_steps,
                            delay_buffer.max(),
                        )
                        prefetched_result = None
                    elif (
                        elapsed_exec_frames > 0
                        and elapsed_exec_frames % interp_stride == 0
                    ):
                        sent_action_frames = max(
                            0,
                            t + 1 - bridge_frame_count,
                        )
                        executed_steps = min(
                            model_steps_from_frame_count(
                                current_result.executed_frame_offset
                                + sent_action_frames
                            ),
                            action_horizon,
                        )
                        root_rel_cum = chunk_root_rel_at_step(
                            current_result,
                            executed_steps,
                        )
                        switched_to_prefetched = True
                        last_chunk_last_pose7 = action_concat_abs[t].copy()
                        logger.info(
                            "switch to GR00T RTC result at model-step boundary: "
                            "elapsed_frames=%d, skipped_steps=%d, executed_steps=%d",
                            elapsed_exec_frames,
                            elapsed_exec_frames // interp_stride,
                            executed_steps,
                        )
                        break

            if action_concat_abs.shape[0] > 0 and not switched_to_prefetched:
                last_chunk_last_pose7 = action_concat_abs[-1].copy()
                root_rel_cum = current_result.next_root_rel.copy()

            logger.info("replay data per frame shape: %s", action_concat_abs[0].shape)

    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    finally:
        sender_thread.stop()
        mocap_queue.clear()
        sender_thread.join(timeout=2.0)
        logger.info("Sender thread stopped.")


if __name__ == "__main__":
    main()

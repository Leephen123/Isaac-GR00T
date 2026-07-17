import os
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from time import sleep

import numpy as np
import tyro
from loop_rate_limiters import RateLimiter
from PIL import Image

from wr.data_res.camera_old import VideoCapture
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
    normalize_quaternion,
    restore_mocap_from_root_relative,
    rotation_6d_to_quaternion,
)
from data_res.utils import (
    CAMERAS_MAP,
    SELECT_11_INDICES,
    get_camera_name,
)
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
    send_fps: float = 50.0
    roll_out: int = 5000
    history_len: int = 50
    camera_warmup_s: float = 2.0
    camera_flush_frames: int = 5
    rtc_action_exec_s: int = 25
    rtc_overlap_steps: int = 25
    rtc_frozen_steps: int = 15
    rtc_ramp_rate: float = 1.0
    rtc_idle_sleep_s: float = 0.01
    root_pose_z: float | None = 1.0
    mocap_cfg: MocapConfig = field(
        default_factory=lambda: MocapConfig(
            domain_id=1, topic_name="MocapUE5G115Topic", depth=4
        )
    )
    body_pose_cfg: BodyPoseConfig = field(
        default_factory=lambda: BodyPoseConfig(
            domain_id=1, topic_name="WR/BodyPose_WR", depth=4
        )
    )
    camera_config: Path = Path("config/camera.yaml")


def extract_mocap_xyz_and_wxyz(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    expected_shape = (MOCAP_NUM_JOINTS, MOCAP_POS_DIM + MOCAP_QUAT_DIM)
    frame = np.asarray(frame, dtype=np.float32)
    if frame.shape != expected_shape:
        raise ValueError(
            f"Expected mocap frame shape {expected_shape}, got {frame.shape}"
        )

    xyz = frame[:, :MOCAP_POS_DIM]
    wxyz = normalize_quaternion(frame[:, MOCAP_POS_DIM:])
    return xyz, wxyz


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

    def get_all(self, poll_interval_s: float = 0.01) -> np.ndarray:
        has_warned = False
        while True:
            with self._lock:
                states = [state.copy() for state in self._queue]

            if states:
                if len(states) < self._queue.maxlen:
                    states = [states[0].copy()] * (
                        self._queue.maxlen - len(states)
                    ) + states
                return np.stack(states, axis=0)

            if not has_warned:
                logger.warning("History queue is empty; waiting for data.")
                has_warned = True
            sleep(poll_interval_s)


def model_action_to_abs_action(
    action_output: np.ndarray,
    init_pose: np.ndarray,
    root_rel_cum: np.ndarray | None = None,
) -> np.ndarray:
    action_output = np.asarray(action_output, dtype=np.float32).reshape(-1, 102)
    root_delta = action_output[:, :3]
    action_without_root_delta = action_output[:, 3:]
    action_mocap_xyz = action_without_root_delta[:, :33].reshape(-1, 11, 3)
    action_mocap_6d = action_without_root_delta[:, 33:].reshape(-1, 11, 6)
    action_mocap = np.concatenate(
        [action_mocap_xyz, action_mocap_6d],
        axis=-1,
    ).reshape(-1, 99)
    action_mocap = np.concatenate([root_delta, action_mocap], axis=-1)
    action_11x9, _ = restore_mocap_from_root_relative(action_mocap, root_rel_cum)

    num_frames = action_11x9.shape[0]
    action_15x9 = np.zeros((num_frames, 15, 9), dtype=np.float32)
    action_15x9[..., 3:9] = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    action_15x9[:, SELECT_11_INDICES, :] = action_11x9

    action_15x7 = rotation_6d_to_quaternion(action_15x9)
    num_frames, num_joints, num_poses = action_15x7.shape
    action_15x7_flat = action_15x7.reshape(num_frames * num_joints, num_poses)
    init_pose_x7_flat = np.tile(init_pose, (num_frames * num_joints, 1))
    return compute_absolute(init_pose_x7_flat, action_15x7_flat).reshape(
        num_frames,
        num_joints,
        num_poses,
    )


def wait_for_body_pose_msg(
    body_pose: BodyPoseSubscriber,
    poll_interval_s: float = 0.01,
) -> WR_GAE_BodyPose_Msg:
    while True:
        msg = body_pose.get_msg()
        if msg is not None:
            return msg
        sleep(poll_interval_s)


class G1Gr00tRtcRunner:
    def __init__(self, config: ClientConfig):
        self._validate_config(config)
        self.config = config
        self.client: server_client.PolicyClient | None = None
        self.body_pose = BodyPoseSubscriber(config.body_pose_cfg)
        self.publisher = MocapUE5G115MsgPublisher(config.mocap_cfg)
        self.state_history = StateHistoryQueue(maxlen=config.history_len)

        camera_name_list = get_camera_name(config.camera_config)
        self.camera_caps = {name: VideoCapture(name) for name in camera_name_list}

        self.root_pose: np.ndarray | None = None
        self.previous_actions: np.ndarray | None = None
        self.previous_action_chunk_rel: np.ndarray | None = None
        self.previous_chunk_start_root_rel: np.ndarray | None = None
        self.previous_chunk_start_step: int | None = None
        self.global_exec_step = 0
        self.update_error: BaseException | None = None
        self.actions_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.client_ready_event = threading.Event()
        self.update_actions_thread = threading.Thread(
            target=self.update_actions_gr00t_rtc,
            daemon=True,
        )

    @staticmethod
    def _validate_config(config: ClientConfig) -> None:
        if config.send_fps <= 0:
            raise ValueError(f"send_fps must be positive, got {config.send_fps}")
        if config.history_len <= 0:
            raise ValueError(
                f"history_len must be positive, got {config.history_len}"
            )
        if config.rtc_action_exec_s <= 0:
            raise ValueError(
                f"rtc_action_exec_s must be positive, got {config.rtc_action_exec_s}"
            )
        if not (
            0
            <= config.rtc_frozen_steps
            <= config.rtc_overlap_steps
        ):
            raise ValueError(
                "GR00T RTC requires 0 <= rtc_frozen_steps <= rtc_overlap_steps, "
                f"got frozen={config.rtc_frozen_steps}, "
                f"overlap={config.rtc_overlap_steps}"
            )
        if not np.isfinite(config.rtc_ramp_rate) or config.rtc_ramp_rate <= 0:
            raise ValueError(
                f"rtc_ramp_rate must be positive, got {config.rtc_ramp_rate}"
            )
        if config.camera_flush_frames <= 0:
            raise ValueError(
                f"camera_flush_frames must be positive, got {config.camera_flush_frames}"
            )
        if config.camera_warmup_s < 0:
            raise ValueError(
                f"camera_warmup_s must be non-negative, got {config.camera_warmup_s}"
            )
        if config.rtc_idle_sleep_s <= 0:
            raise ValueError(
                f"rtc_idle_sleep_s must be positive, got {config.rtc_idle_sleep_s}"
            )

    def start(self) -> None:
        initial_body_pose_msg = wait_for_body_pose_msg(self.body_pose)
        self.state_history.put(initial_body_pose_msg)
        self.root_pose = self.body_pose.msg_to_root_pose(initial_body_pose_msg)
        if self.config.root_pose_z is not None:
            self.root_pose[2] = float(self.config.root_pose_z)
        sleep(self.config.camera_warmup_s)
        self.update_actions_thread.start()
        while not self.client_ready_event.wait(timeout=self.config.rtc_idle_sleep_s):
            if self.update_error is not None:
                raise RuntimeError(
                    "Failed to initialize GR00T RTC policy client"
                ) from self.update_error

    def stop(self) -> None:
        self.stop_event.set()
        self.client_ready_event.set()
        if self.update_actions_thread.is_alive():
            self.update_actions_thread.join(timeout=2.0)
        for camera_cap in self.camera_caps.values():
            camera_cap.release()

    def capture_frames(self) -> dict[str, np.ndarray]:
        frames = {}
        for name, camera_cap in self.camera_caps.items():
            for _ in range(self.config.camera_flush_frames):
                ok, frame = camera_cap.read()
                if not ok or frame is None:
                    raise RuntimeError(f"Failed to read camera frame from {name}")
            frames[name] = frame
        return frames

    def build_observation(self) -> dict:
        observation = build_observation_from_msg_with_history(
            self.state_history.get_all(),
            self.config.task_description,
            frame=self.capture_frames(),
        )
        observation.update(self._gr00t_rtc_inputs())
        return observation

    def get_actions(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
        with self.actions_lock:
            chunk_start_step = (
                0
                if self.previous_chunk_start_step is None
                else self.previous_chunk_start_step + self.config.rtc_action_exec_s
            )
            if self.global_exec_step < chunk_start_step:
                raise RuntimeError(
                    "GR00T RTC inference started before the next chunk boundary: "
                    f"global={self.global_exec_step}, chunk_start={chunk_start_step}"
                )
            previous_action_chunk_rel = (
                None
                if self.previous_action_chunk_rel is None
                else self.previous_action_chunk_rel.copy()
            )
            previous_chunk_start_root_rel = (
                None
                if self.previous_chunk_start_root_rel is None
                else self.previous_chunk_start_root_rel.copy()
            )
            previous_chunk_start_step = self.previous_chunk_start_step

        chunk_start_root_rel = self._chunk_start_root_rel(
            chunk_start_step,
            previous_chunk_start_step,
            previous_action_chunk_rel,
            previous_chunk_start_root_rel,
        )
        if self.client is None:
            raise RuntimeError("Policy client is not initialized")
        response = self.client.get_action(self.build_observation())
        action_chunk_rel = np.asarray(
            response[0]["mocap"][0], dtype=np.float32
        ).reshape(-1, 102)
        self._validate_action_horizon(action_chunk_rel.shape[0])

        if self.root_pose is None:
            raise RuntimeError("root_pose is not initialized")
        actions = model_action_to_abs_action(
            action_chunk_rel,
            self.root_pose,
            chunk_start_root_rel,
        )

        delay_steps = self._current_global_exec_step() - chunk_start_step
        if delay_steps >= action_chunk_rel.shape[0]:
            raise RuntimeError(
                "Discard stale action chunk: "
                f"delay={delay_steps}, horizon={action_chunk_rel.shape[0]}"
            )
        if delay_steps > self.config.rtc_frozen_steps:
            logger.warning(
                "Inference delay exceeds rtc_frozen_steps: delay=%d, frozen=%d",
                delay_steps,
                self.config.rtc_frozen_steps,
            )
        return (
            actions,
            action_chunk_rel,
            chunk_start_root_rel,
            chunk_start_step,
            delay_steps,
        )

    def update_actions_gr00t_rtc(self) -> None:
        try:
            self._initialize_policy_client()
        except Exception as exc:
            self.update_error = exc
            self.stop_event.set()
            self.client_ready_event.set()
            logger.exception("Failed to initialize GR00T RTC policy client.")
            return

        while not self.stop_event.is_set():
            with self.actions_lock:
                should_update = (
                    self.previous_chunk_start_step is None
                    or self.global_exec_step
                    >= self.previous_chunk_start_step
                    + self.config.rtc_action_exec_s
                )

            if should_update:
                try:
                    (
                        actions,
                        action_chunk_rel,
                        chunk_start_root_rel,
                        chunk_start_step,
                        delay_steps,
                    ) = self.get_actions()
                except Exception as exc:
                    self.update_error = exc
                    self.stop_event.set()
                    logger.exception("GR00T RTC action update failed.")
                    return

                with self.actions_lock:
                    self.previous_actions = actions.copy()
                    self.previous_action_chunk_rel = action_chunk_rel.copy()
                    self.previous_chunk_start_root_rel = chunk_start_root_rel.copy()
                    self.previous_chunk_start_step = chunk_start_step
                logger.info(
                    "Installed GR00T RTC chunk: start=%d, delay=%d, horizon=%d",
                    chunk_start_step,
                    delay_steps,
                    actions.shape[0],
                )

            sleep(self.config.rtc_idle_sleep_s)

    def _initialize_policy_client(self) -> None:
        self.client = server_client.PolicyClient(
            host=self.config.host,
            port=self.config.port,
            timeout_ms=self.config.timeout_ms,
            api_token=self.config.api_token,
        )
        print("Waiting for gr00t server to ping")
        if not self.client.ping():
            raise ConnectionError("Failed to connect to the gr00t server")
        self.client.reset()
        print("Server is alive!")
        self.client_ready_event.set()

    def action_execution_gr00t_rtc(self, roll_out_len: int) -> None:
        rate = RateLimiter(frequency=self.config.send_fps)
        frames_sent = 0

        while frames_sent < roll_out_len:
            if self.update_error is not None:
                raise RuntimeError(
                    "GR00T RTC update thread failed"
                ) from self.update_error
            if self.stop_event.is_set():
                break

            with self.actions_lock:
                actions = self.previous_actions
                chunk_start_step = self.previous_chunk_start_step
                if actions is None or actions.shape[0] == 0:
                    sent_action = False
                else:
                    if chunk_start_step is None:
                        raise RuntimeError(
                            "Action chunk is missing its chunk start step"
                        )
                    action_index = max(
                        0,
                        min(
                            self.global_exec_step - chunk_start_step,
                            actions.shape[0] - 1,
                        ),
                    )
                    mocap_frame = actions[action_index]
                    xyz, wxyz = extract_mocap_xyz_and_wxyz(mocap_frame)
                    self.publisher.send_msg(
                        fps=self.config.send_fps,
                        xyz=xyz,
                        wxyz=wxyz,
                    )
                    body_pose_msg = self.body_pose.get_msg()
                    if body_pose_msg is not None:
                        self.state_history.put(body_pose_msg)
                    self.global_exec_step += 1
                    sent_action = True

            if not sent_action:
                rate.sleep()
                continue

            frames_sent += 1
            rate.sleep()

    def _current_global_exec_step(self) -> int:
        with self.actions_lock:
            return self.global_exec_step

    def _validate_action_horizon(self, action_horizon: int) -> None:
        if self.config.rtc_overlap_steps > action_horizon:
            raise ValueError(
                "rtc_overlap_steps must not exceed the action horizon, "
                f"got overlap={self.config.rtc_overlap_steps}, H={action_horizon}"
            )
        expected_action_exec_s = action_horizon - self.config.rtc_overlap_steps
        if self.config.rtc_action_exec_s != expected_action_exec_s:
            raise ValueError(
                "GR00T RTC requires rtc_action_exec_s == "
                "action_horizon - rtc_overlap_steps, "
                f"got action_exec={self.config.rtc_action_exec_s}, "
                f"overlap={self.config.rtc_overlap_steps}, H={action_horizon}"
            )

    def _gr00t_rtc_inputs(self) -> dict[str, np.ndarray]:
        return {
            "rtc_overlap_steps": np.asarray(
                self.config.rtc_overlap_steps, dtype=np.int32
            ),
            "rtc_frozen_steps": np.asarray(
                self.config.rtc_frozen_steps, dtype=np.int32
            ),
            "rtc_ramp_rate": np.asarray(
                self.config.rtc_ramp_rate, dtype=np.float32
            ),
        }

    @staticmethod
    def _chunk_start_root_rel(
        chunk_start_step: int,
        previous_chunk_start_step: int | None,
        previous_action_chunk_rel: np.ndarray | None,
        previous_chunk_start_root_rel: np.ndarray | None,
    ) -> np.ndarray:
        if (
            previous_chunk_start_step is None
            or previous_action_chunk_rel is None
            or previous_chunk_start_root_rel is None
        ):
            return np.zeros(3, dtype=np.float32)

        executed_steps = chunk_start_step - previous_chunk_start_step
        if executed_steps < 0:
            raise ValueError(
                "Chunk start step moved backwards: "
                f"current={chunk_start_step}, "
                f"previous={previous_chunk_start_step}"
            )
        delta_count = min(executed_steps, previous_action_chunk_rel.shape[0])
        root_delta = previous_action_chunk_rel[:delta_count, :3].sum(axis=0)
        return previous_chunk_start_root_rel + root_delta


def main() -> None:
    config = tyro.cli(ClientConfig)
    runner = G1Gr00tRtcRunner(config)
    try:
        runner.start()
        runner.action_execution_gr00t_rtc(config.roll_out)
    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    finally:
        runner.stop()
        logger.info("GR00T RTC inference stopped.")


if __name__ == "__main__":
    main()

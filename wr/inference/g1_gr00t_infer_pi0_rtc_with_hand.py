import os
import sys
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from time import sleep

import numpy as np
import tyro
from loop_rate_limiters import RateLimiter
from PIL import Image

root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))
sys.path.append(os.getcwd())

from wr.data_res.camera_old import VideoCapture
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
    interpolate_pose7,
    normalize_quaternion,
    quaternion_to_rotation_6d,
    restore_mocap_from_root_relative,
    rotation_6d_to_quaternion,
)
from data_res.utils import CAMERAS_MAP, SELECT_11_INDICES, get_camera_name
from serve_res.gr00t import server_client

logger = get_logger(__name__)

HAND_ACTION_DIM = 12
BODY_STATE_DIM = 35
HAND_STATE_DIM = 12
MODEL_ACTION_DIM_WITH_HAND = 114


@dataclass
class ClientConfig:
    host: str = "192.168.123.165"
    port: int = 9002
    timeout_ms: int = 15000
    api_token: str | None = None
    task_description: str = (
        "pick up the water to bowl and kitchen sink"
    )
    send_fps: float = 120
    history_len: int = 50
    roll_out: int = 20000
    rtc_action_exec_s: int = 35
    rtc_init_delay_frames: int = 15
    rtc_delay_buffer_len: int = 3
    rtc_idle_sleep_s: float = 0.01
    rtc_beta: float = 5.0
    enable_action_interp: bool = True
    interp_num: int = 2
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


def root_pose_from_msg(msg: WR_GAE_BodyPose_Msg_V2) -> np.ndarray:
    root_xyz = np.asarray(msg.robot_xpos[36:39], dtype=np.float32)
    root_wxyz = np.asarray(msg.robot_rootquat, dtype=np.float32)
    return np.concatenate([root_xyz, root_wxyz], axis=0)


def body_hand_state_from_msg(
    msg: WR_GAE_BodyPose_Msg_V2,
    hand_state: np.ndarray,
) -> np.ndarray:
    q_np = np.asarray(msg.robot_qpos[:29], dtype=np.float32)
    if q_np.shape != (29,):
        raise ValueError(f"Expected robot_qpos[:29] shape (29,), got {q_np.shape}")

    imu_np = np.asarray(msg.robot_rootquat, dtype=np.float32)
    imu_np = compute_imu_relative(imu_np[None, :], imu_np[None, :])
    imu_np = quaternion_to_rotation_6d(imu_np)[0]

    hand_np = np.asarray(hand_state, dtype=np.float32).reshape(-1)
    if hand_np.shape != (HAND_STATE_DIM,):
        raise ValueError(
            f"Expected hand state shape ({HAND_STATE_DIM},), got {hand_np.shape}"
        )

    body_state = np.concatenate([imu_np, q_np], axis=0)
    if body_state.shape != (BODY_STATE_DIM,):
        raise ValueError(
            f"Expected body state shape ({BODY_STATE_DIM},), got {body_state.shape}"
        )
    return np.concatenate([body_state, hand_np], axis=0).astype(np.float32)


class StateHistoryQueue:
    def __init__(self, maxlen: int = 50):
        self._lock = threading.Lock()
        self._queue = deque(maxlen=maxlen)

    def put(self, msg: WR_GAE_BodyPose_Msg_V2, hand_state: np.ndarray) -> bool:
        state = body_hand_state_from_msg(msg, hand_state)
        with self._lock:
            self._queue.append(state)
            return True

    def get_all(self, poll_interval: float = 0.01) -> np.ndarray:
        has_warned = False
        maxlen = self._queue.maxlen

        while True:
            states = None
            with self._lock:
                if self._queue:
                    states = [state.copy() for state in self._queue]

            if states is not None:
                if len(states) < maxlen:
                    states = [states[0].copy()] * (maxlen - len(states)) + states
                return np.stack(states, axis=0)

            if not has_warned:
                logger.warning("History queue is empty; waiting for data to be filled.")
                has_warned = True
            sleep(poll_interval)


class DelayBuffer:
    def __init__(self, initial_delay_frames: int, maxlen: int = 10):
        if maxlen <= 0:
            raise ValueError(f"maxlen must be positive, got {maxlen}")
        self._lock = threading.Lock()
        self._queue = deque([max(int(initial_delay_frames), 0)], maxlen=maxlen)

    def add(self, delay_frames: int) -> None:
        with self._lock:
            self._queue.append(max(int(delay_frames), 0))

    def max(self) -> int:
        with self._lock:
            return max(self._queue)


def load_image_from_array(
    image_input: np.ndarray,
    target_size: tuple[int, int] = (256, 256),
) -> np.ndarray:
    image_arr = np.asarray(image_input)
    if image_arr.ndim != 3 or image_arr.shape[-1] != 3:
        raise ValueError(f"Expected frame shape (H, W, 3), got {image_arr.shape}")
    image = Image.fromarray(image_arr.astype(np.uint8)[:, :, ::-1]).convert("RGB")
    return np.asarray(image.resize(target_size, Image.BILINEAR))[None, None, ...]


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
    if cur_chunk_hand.ndim != 2 or cur_chunk_hand.shape[1] != HAND_ACTION_DIM:
        raise ValueError(
            f"Expected cur_chunk_hand shape (T,{HAND_ACTION_DIM}), "
            f"got {cur_chunk_hand.shape}"
        )
    if cur_chunk_pose7.shape[0] == 0:
        raise ValueError("Cannot install empty action chunk")
    if cur_chunk_pose7.shape[0] != cur_chunk_hand.shape[0]:
        raise ValueError(
            "Mocap and hand action lengths do not match before interpolation: "
            f"{cur_chunk_pose7.shape[0]} vs {cur_chunk_hand.shape[0]}"
        )
    if not enable_interp or num_interp <= 0:
        return cur_chunk_pose7, cur_chunk_hand, 0

    prepended_bridge_frames = 0
    source_pose7 = cur_chunk_pose7
    source_hand = cur_chunk_hand
    if allow_bridge and prev_last_pose7 is not None and prev_last_hand is not None:
        prev_last_pose7 = np.asarray(prev_last_pose7, dtype=np.float32)
        prev_last_hand = np.asarray(prev_last_hand, dtype=np.float32)
        if prev_last_pose7.shape != (15, 7):
            raise ValueError(
                f"Expected prev_last_pose7 shape (15,7), got {prev_last_pose7.shape}"
            )
        if prev_last_hand.shape != (HAND_ACTION_DIM,):
            raise ValueError(
                f"Expected prev_last_hand shape ({HAND_ACTION_DIM},), "
                f"got {prev_last_hand.shape}"
            )
        source_pose7 = np.concatenate(
            [prev_last_pose7[None, ...], cur_chunk_pose7],
            axis=0,
        )
        source_hand = np.concatenate(
            [prev_last_hand[None, ...], cur_chunk_hand],
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
    if hand_joint_seq.ndim != 2 or hand_joint_seq.shape[1] != HAND_ACTION_DIM:
        raise ValueError(
            f"Expected hand joint shape (T,{HAND_ACTION_DIM}), got {hand_joint_seq.shape}"
        )
    if hand_joint_seq.shape[0] <= 1 or num_interp <= 0:
        return hand_joint_seq
    steps = np.linspace(0.0, 1.0, num_interp + 2, dtype=np.float32)[:-1]
    out = (
        (1.0 - steps[None, :, None]) * hand_joint_seq[:-1, None, :]
        + steps[None, :, None] * hand_joint_seq[1:, None, :]
    )
    return np.concatenate(
        [out.reshape(-1, HAND_ACTION_DIM), hand_joint_seq[-1:]],
        axis=0,
    )


def at_model_boundary_from_frame_count(frame_count: int, interp_stride: int) -> bool:
    return frame_count > 0 and (frame_count - 1) % interp_stride == 0


def model_steps_from_frame_count(frame_count: int, interp_stride: int) -> int:
    if frame_count <= 0:
        return 0
    return (frame_count + interp_stride - 1) // interp_stride


def action_frame_offset_after_steps(steps: int, interp_stride: int) -> int:
    if steps <= 0:
        return 0
    return (steps - 1) * interp_stride + 1


def build_observation_from_msg_with_hand_history(
    history_state: np.ndarray,
    task_description: str,
    frame: dict[str, np.ndarray] | None,
) -> dict:
    history_state = np.asarray(history_state, dtype=np.float32)
    expected_dim = BODY_STATE_DIM + HAND_STATE_DIM
    if history_state.ndim != 2 or history_state.shape[1] != expected_dim:
        raise ValueError(
            f"Expected hand history state shape (T, {expected_dim}), "
            f"got {history_state.shape}"
        )

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


def model_action_to_abs_action(
    action_output: np.ndarray,
    init_pose: np.ndarray,
    root_rel_cum: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    action_output = np.asarray(action_output, dtype=np.float32)
    if action_output.size % MODEL_ACTION_DIM_WITH_HAND != 0:
        raise ValueError(
            "Expected model action size divisible by "
            f"{MODEL_ACTION_DIM_WITH_HAND}, got {action_output.shape}"
        )
    action_output = action_output.reshape(-1, MODEL_ACTION_DIM_WITH_HAND)
    if not np.isfinite(action_output).all():
        raise ValueError("Model returned non-finite actions")
    if not np.any(np.abs(action_output) > 1e-6):
        logger.warning("Model returned an all-zero action chunk")

    root_delta = action_output[:, :3]
    action_without_root_delta = action_output[:, 3:102]
    action_hand = action_output[:, 102:]
    if action_hand.shape[1] != HAND_ACTION_DIM:
        raise ValueError(
            f"Expected hand action dim {HAND_ACTION_DIM}, got {action_hand.shape}"
        )

    action_mocap_xyz = action_without_root_delta[:, :33].reshape(-1, 11, 3)
    action_mocap_6d = action_without_root_delta[:, 33:].reshape(-1, 11, 6)
    action_mocap = np.concatenate(
        [action_mocap_xyz, action_mocap_6d],
        axis=-1,
    ).reshape(-1, 99)
    action_mocap = np.concatenate([root_delta, action_mocap], axis=-1)

    action_11x9, next_root_rel_cum = restore_mocap_from_root_relative(
        action_mocap,
        root_rel_cum,
    )

    num_frames = action_11x9.shape[0]
    action_15x9 = np.zeros((num_frames, 15, 9), dtype=np.float32)
    action_15x9[..., 3:9] = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    action_15x9[:, SELECT_11_INDICES, :] = action_11x9

    action_15x7 = rotation_6d_to_quaternion(action_15x9)
    num_frames, num_joints, num_poses = action_15x7.shape
    action_15x7_flat = action_15x7.reshape(num_frames * num_joints, num_poses)
    init_pose_x7_flat = np.tile(init_pose, (num_frames * num_joints, 1))

    abs_action = compute_absolute(init_pose_x7_flat, action_15x7_flat).reshape(
        num_frames,
        num_joints,
        num_poses,
    )
    if not np.isfinite(abs_action).all() or not np.isfinite(next_root_rel_cum).all():
        raise ValueError("Restored mocap action contains non-finite values")
    return abs_action, action_hand.astype(np.float32), next_root_rel_cum


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


class G1Pi0RtcWithHandRunner:
    def __init__(self, config: ClientConfig):
        self._validate_config(config)

        self.config = config
        self.interp_stride = (
            config.interp_num + 1 if config.enable_action_interp else 1
        )
        self.client: server_client.PolicyClient | None = None
        self.body_pose = BodyPoseSubscriberV3(config.body_pose_cfg)
        self.hand_subscriber = MocapUEHandSubscriber()
        self.publisher = MocapUE5G115MsgPublisher(config.mocap_cfg)
        self.hand_publisher = MocapUEHandPublisher()
        self.state_history = StateHistoryQueue(maxlen=config.history_len)

        camera_name_list = get_camera_name(config.camera_config)
        self.camera_caps = {name: VideoCapture(name) for name in camera_name_list}

        self.root_pose: np.ndarray | None = None
        self.previous_actions: np.ndarray | None = None
        self.previous_hand_actions: np.ndarray | None = None
        self.previous_action_chunk_rel: np.ndarray | None = None
        self.previous_chunk_start_root_rel: np.ndarray | None = None
        self.previous_inference_start_step: int | None = None
        self.current_frame_cursor = 0
        self.current_action_frame_offset = 0
        self.current_prepended_bridge_frames = 0
        self.global_exec_step = 0
        self.global_action_frame_count = 0
        self.last_sent_pose: np.ndarray | None = None
        self.last_sent_hand: np.ndarray | None = None
        self.force_normal_next = False
        self.update_error: BaseException | None = None
        self.actions_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.client_ready_event = threading.Event()
        self.update_actions_thread = threading.Thread(
            target=self.update_actions_rtc,
            daemon=True,
        )
        self.delay_buffer = DelayBuffer(
            initial_delay_frames=config.rtc_init_delay_frames,
            maxlen=config.rtc_delay_buffer_len,
        )

    @staticmethod
    def _validate_config(config: ClientConfig) -> None:
        if config.send_fps <= 0:
            raise ValueError(f"send_fps must be positive, got {config.send_fps}")
        if config.history_len <= 0:
            raise ValueError(f"history_len must be positive, got {config.history_len}")
        if config.rtc_action_exec_s <= 0:
            raise ValueError(
                f"rtc_action_exec_s must be positive, got {config.rtc_action_exec_s}"
            )
        if config.rtc_init_delay_frames < 0:
            raise ValueError(
                "rtc_init_delay_frames must be non-negative, "
                f"got {config.rtc_init_delay_frames}"
            )
        if config.rtc_delay_buffer_len <= 0:
            raise ValueError(
                "rtc_delay_buffer_len must be positive, "
                f"got {config.rtc_delay_buffer_len}"
            )
        if config.rtc_idle_sleep_s <= 0:
            raise ValueError(
                f"rtc_idle_sleep_s must be positive, got {config.rtc_idle_sleep_s}"
            )
        if not np.isfinite(config.rtc_beta) or config.rtc_beta <= 0:
            raise ValueError(f"rtc_beta must be positive, got {config.rtc_beta}")
        if config.interp_num < 0:
            raise ValueError(f"interp_num must be non-negative, got {config.interp_num}")

    def initialize_policy_client(self) -> None:
        self.client = server_client.PolicyClient(
            host=self.config.host,
            port=self.config.port,
            timeout_ms=self.config.timeout_ms,
            api_token=self.config.api_token,
        )
        print("Waiting for gr00t server to ping")
        if self.client.ping():
            print("Server is alive!")
            self.client.reset()
            self.client_ready_event.set()
            return
        raise ConnectionError("Failed to connect to the gr00t server")

    def start(self) -> None:
        sleep(2)
        initial_msg = wait_for_body_pose_msg(self.body_pose)
        initial_hand = wait_for_hand_state_msg(self.hand_subscriber)
        while True:
            self.root_pose = self.body_pose.get_root_pose()
            if self.root_pose is not None:
                print(f"root pose received!")
                break
        if self.config.root_pose_z is not None:
            self.root_pose[2] = float(self.config.root_pose_z)
        self.state_history.put(initial_msg, initial_hand)

        self.update_actions_thread.start()
        while not self.client_ready_event.wait(timeout=self.config.rtc_idle_sleep_s):
            if self.update_error is not None:
                raise RuntimeError(
                    "Failed to initialize RTC policy client"
                ) from self.update_error
        if self.update_error is not None:
            raise RuntimeError(
                "Failed to initialize RTC policy client"
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
            ok, frame = camera_cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"Failed to read camera frame from {name}")
            frames[name] = frame
        return frames

    def build_observation(
        self,
        state_history: np.ndarray,
        frames: dict[str, np.ndarray],
    ) -> dict:
        return build_observation_from_msg_with_hand_history(
            state_history,
            self.config.task_description,
            frame=frames,
        )

    def get_actions(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
        frames = self.capture_frames()
        with self.actions_lock:
            inference_start_step = self.global_exec_step
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
            previous_inference_start_step = self.previous_inference_start_step
            force_normal_next = self.force_normal_next
            state_history = self.state_history.get_all()

        chunk_start_root_rel = self._chunk_start_root_rel(
            inference_start_step=inference_start_step,
            previous_inference_start_step=previous_inference_start_step,
            previous_action_chunk_rel=previous_action_chunk_rel,
            previous_chunk_start_root_rel=previous_chunk_start_root_rel,
        )

        observation = self.build_observation(state_history, frames)
        rtc_inputs = None
        if not force_normal_next:
            rtc_inputs = self._build_rtc_inputs(
                previous_inference_start_step=previous_inference_start_step,
                inference_start_step=inference_start_step,
                previous_action_horizon=(
                    None
                    if previous_action_chunk_rel is None
                    else previous_action_chunk_rel.shape[0]
                ),
            )
        if rtc_inputs is not None:
            observation.update(rtc_inputs)

        if self.client is None:
            raise RuntimeError("Policy client is not initialized")
        response = self.client.get_action(observation)
        action_chunk_rel = np.asarray(
            response[0]["mocap"][0], dtype=np.float32
        ).reshape(-1, MODEL_ACTION_DIM_WITH_HAND)

        if self.root_pose is None:
            raise RuntimeError("root_pose is not initialized")

        action_chunk_abs, hand_actions, _ = model_action_to_abs_action(
            action_chunk_rel,
            self.root_pose,
            chunk_start_root_rel,
        )
        if action_chunk_abs.shape[0] != hand_actions.shape[0]:
            raise ValueError(
                "Mocap and hand action lengths do not match: "
                f"{action_chunk_abs.shape[0]} vs {hand_actions.shape[0]}"
            )

        measured_delay_steps = max(
            self._current_global_exec_step() - inference_start_step, 0
        )
        if measured_delay_steps >= action_chunk_rel.shape[0]:
            raise RuntimeError(
                "Discard stale action chunk: "
                f"delay={measured_delay_steps}, horizon={action_chunk_rel.shape[0]}"
            )

        return (
            action_chunk_abs,
            hand_actions,
            action_chunk_rel,
            chunk_start_root_rel,
            inference_start_step,
            measured_delay_steps,
        )

    def update_actions_rtc(self) -> None:
        try:
            self.initialize_policy_client()
        except Exception as exc:
            self.update_error = exc
            self.stop_event.set()
            self.client_ready_event.set()
            logger.exception("Failed to initialize RTC policy client.")
            return

        while not self.stop_event.is_set():
            with self.actions_lock:
                should_update = (
                    self.previous_inference_start_step is None
                    or self.global_exec_step
                    >= self.previous_inference_start_step
                    + self.config.rtc_action_exec_s
                )

            if should_update:
                try:
                    (
                        actions,
                        hand_actions,
                        action_chunk_rel,
                        chunk_start_root_rel,
                        inference_start_step,
                        delay_steps,
                    ) = self.get_actions()
                except Exception as exc:
                    if self.previous_actions is None:
                        self.update_error = exc
                        self.stop_event.set()
                        logger.exception("Initial action update failed.")
                        return
                    logger.exception(
                        "RTC action update failed; resetting server cache and retrying "
                        "with normal inference."
                    )
                    try:
                        if self.client is None:
                            raise RuntimeError("Policy client is not initialized")
                        self.client.reset()
                    except Exception as reset_exc:
                        self.update_error = reset_exc
                        self.stop_event.set()
                        logger.exception("Failed to reset RTC policy cache.")
                        return
                    with self.actions_lock:
                        self.force_normal_next = True
                    sleep(self.config.rtc_idle_sleep_s)
                    continue

                (
                    action_frames,
                    hand_frames,
                    action_frame_offset,
                    prepended_bridge_frames,
                ) = self._prepare_action_frames(
                    actions=actions,
                    hand_actions=hand_actions,
                    delay_steps=delay_steps,
                )

                with self.actions_lock:
                    if self.previous_inference_start_step is not None:
                        self.delay_buffer.add(delay_steps)
                    self.previous_actions = action_frames.copy()
                    self.previous_hand_actions = hand_frames.copy()
                    self.previous_action_chunk_rel = action_chunk_rel.copy()
                    self.previous_chunk_start_root_rel = chunk_start_root_rel.copy()
                    self.previous_inference_start_step = inference_start_step
                    self.current_frame_cursor = 0
                    self.current_action_frame_offset = action_frame_offset
                    self.current_prepended_bridge_frames = prepended_bridge_frames
                    self.force_normal_next = False

            sleep(self.config.rtc_idle_sleep_s)

    def _prepare_action_frames(
        self,
        actions: np.ndarray,
        hand_actions: np.ndarray,
        delay_steps: int,
    ) -> tuple[np.ndarray, np.ndarray, int, int]:
        with self.actions_lock:
            prev_last_pose = (
                None if self.last_sent_pose is None else self.last_sent_pose.copy()
            )
            prev_last_hand = (
                None if self.last_sent_hand is None else self.last_sent_hand.copy()
            )

        full_frames, full_hands, prepended_bridge_frames = build_action_frames(
            prev_last_pose7=prev_last_pose,
            prev_last_hand=prev_last_hand,
            cur_chunk_pose7=actions,
            cur_chunk_hand=hand_actions,
            enable_interp=self.config.enable_action_interp,
            allow_bridge=True,
            num_interp=self.config.interp_num,
        )

        action_frame_offset = action_frame_offset_after_steps(
            delay_steps,
            self.interp_stride,
        )
        action_frame_count = full_frames.shape[0] - prepended_bridge_frames
        if action_frame_offset < 0 or action_frame_offset >= action_frame_count:
            raise RuntimeError(
                "Discard stale interpolated action chunk: "
                f"delay_steps={delay_steps}, "
                f"action_frame_offset={action_frame_offset}, "
                f"action_frames={action_frame_count}"
            )

        if prepended_bridge_frames > 0 and action_frame_offset == 0:
            frames = full_frames
            hand_frames = full_hands
        else:
            frames = full_frames[prepended_bridge_frames + action_frame_offset:]
            hand_frames = full_hands[prepended_bridge_frames + action_frame_offset:]
            prepended_bridge_frames = 0

        return (
            frames.astype(np.float32),
            hand_frames.astype(np.float32),
            action_frame_offset,
            prepended_bridge_frames,
        )

    def action_execution_rtc(self, roll_out_len: int) -> None:
        rate = RateLimiter(frequency=self.config.send_fps)
        frames_sent = 0

        while frames_sent < roll_out_len:
            if self.update_error is not None:
                raise RuntimeError("RTC update thread failed") from self.update_error
            if self.stop_event.is_set():
                break

            with self.actions_lock:
                actions = self.previous_actions
                hand_actions = self.previous_hand_actions
                if (
                    actions is None
                    or hand_actions is None
                    or actions.shape[0] == 0
                    or hand_actions.shape[0] == 0
                ):
                    sent_action = False
                    should_put_state = False
                else:
                    if self.previous_inference_start_step is None:
                        raise RuntimeError(
                            "Action chunk is missing its inference start step"
                        )
                    action_index = min(self.current_frame_cursor, actions.shape[0] - 1)
                    hand_index = min(self.current_frame_cursor, hand_actions.shape[0] - 1)
                    mocap_frame = actions[action_index].copy()
                    hand_frame = hand_actions[hand_index].copy()
                    is_holding_last_frame = self.current_frame_cursor >= actions.shape[0]
                    should_put_state = False

                    if not is_holding_last_frame:
                        is_action_frame = (
                            self.current_frame_cursor
                            >= self.current_prepended_bridge_frames
                        )
                        self.current_frame_cursor += 1
                        if is_action_frame:
                            self.global_action_frame_count += 1
                            sent_action_frames = max(
                                0,
                                self.current_frame_cursor
                                - self.current_prepended_bridge_frames,
                            )
                            full_frame_count = (
                                self.current_action_frame_offset
                                + sent_action_frames
                            )
                            should_put_state = at_model_boundary_from_frame_count(
                                full_frame_count,
                                self.interp_stride,
                            )
                            if should_put_state:
                                self.global_exec_step += 1

                    self.last_sent_pose = mocap_frame.copy()
                    self.last_sent_hand = hand_frame.copy()
                    sent_action = True

            if not sent_action:
                rate.sleep()
                continue

            xyz, wxyz = extract_mocap_xyz_and_wxyz(mocap_frame)
            self.publisher.send_msg(
                fps=self.config.send_fps,
                xyz=xyz,
                wxyz=wxyz,
            )
            self.hand_publisher.send(hand_frame)
            if should_put_state:
                self.put_current_body_pose_to_history()

            frames_sent += 1
            rate.sleep()

    def put_current_body_pose_to_history(self) -> None:
        msg = self.body_pose.get_msg()
        hand_state = self.hand_subscriber.get_state()
        if msg is None or hand_state is None:
            logger.warning(
                "Skip state history update: body pose or hand state is unavailable."
            )
            return
        self.state_history.put(msg, hand_state)

    def _current_global_exec_step(self) -> int:
        with self.actions_lock:
            return self.global_exec_step

    @staticmethod
    def _chunk_start_root_rel(
        inference_start_step: int,
        previous_inference_start_step: int | None,
        previous_action_chunk_rel: np.ndarray | None,
        previous_chunk_start_root_rel: np.ndarray | None,
    ) -> np.ndarray:
        if (
            previous_inference_start_step is None
            or previous_action_chunk_rel is None
            or previous_chunk_start_root_rel is None
        ):
            return np.zeros(3, dtype=np.float32)

        previous_action_chunk_rel = np.asarray(
            previous_action_chunk_rel, dtype=np.float32
        )
        previous_chunk_start_root_rel = np.asarray(
            previous_chunk_start_root_rel, dtype=np.float32
        )
        if (
            previous_action_chunk_rel.ndim != 2
            or previous_action_chunk_rel.shape[1] != MODEL_ACTION_DIM_WITH_HAND
            or previous_action_chunk_rel.shape[0] == 0
        ):
            raise ValueError(
                "Expected previous decoded actions with shape "
                f"(H, {MODEL_ACTION_DIM_WITH_HAND}), "
                f"got {previous_action_chunk_rel.shape}"
            )
        if previous_chunk_start_root_rel.shape != (3,):
            raise ValueError(
                "Expected previous chunk root anchor with shape (3,), "
                f"got {previous_chunk_start_root_rel.shape}"
            )

        executed_steps = inference_start_step - previous_inference_start_step
        if executed_steps < 0:
            raise ValueError(
                "Inference start step moved backwards: "
                f"current={inference_start_step}, previous={previous_inference_start_step}"
            )

        delta_count = min(executed_steps, previous_action_chunk_rel.shape[0])
        root_delta = previous_action_chunk_rel[:delta_count, :3].sum(axis=0)
        return previous_chunk_start_root_rel + root_delta

    def _build_rtc_inputs(
        self,
        previous_inference_start_step: int | None,
        inference_start_step: int,
        previous_action_horizon: int | None,
    ) -> dict[str, np.ndarray] | None:
        if previous_inference_start_step is None or previous_action_horizon is None:
            return None

        horizon = previous_action_horizon
        executed_steps = inference_start_step - previous_inference_start_step
        estimated_delay_steps = self.delay_buffer.max()

        if not (
            estimated_delay_steps <= executed_steps
            and executed_steps <= horizon - estimated_delay_steps
        ):
            logger.warning(
                "RTC infeasible; falling back to normal inference: H=%d, s=%d, d=%d",
                horizon,
                executed_steps,
                estimated_delay_steps,
            )
            return None

        return {
            "delay_frames": np.asarray(estimated_delay_steps, dtype=np.int32),
            "action_executed_steps": np.asarray(executed_steps, dtype=np.int32),
            "rtc_beta": np.asarray(self.config.rtc_beta, dtype=np.float32),
        }


def main() -> None:
    config = tyro.cli(ClientConfig)
    runner = G1Pi0RtcWithHandRunner(config)
    try:
        runner.start()
        runner.action_execution_rtc(config.roll_out)
    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    finally:
        runner.stop()
        logger.info("PI0 RTC inference with hand stopped.")


if __name__ == "__main__":
    main()

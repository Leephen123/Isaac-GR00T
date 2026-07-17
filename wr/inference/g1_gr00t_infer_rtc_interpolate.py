import sys
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from time import sleep

import numpy as np
import tyro
from loop_rate_limiters import RateLimiter

from wr.data_res.camera_old import VideoCapture
from data_res.dds import (
    BODY_POSE_Q_SIZE,
    BODY_POSE_WXYZ_SIZE,
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
from data_res.utils import (
    SELECT_11_INDICES,
    build_observation_from_msg_rtc,
    get_camera_name,
)
from inference.utils import SIM_INIT_POSE
from serve_res.gr00t import server_client

logger = get_logger(__name__)


@dataclass
class ClientConfig:
    host: str = "192.168.123.163"
    port: int = 9002
    timeout_ms: int = 15000
    task_description: str = "Clean the items on the desk and put the doll on the left sofa"
    send_fps: float = 50.0
    history_len: int = 50
    num_interp: int = 20
    use_interpolate: bool = True
    roll_out: int = 5000
    rtc_action_exec_s: int = 10
    rtc_init_delay_frames: int = 2
    rtc_delay_buffer_len: int = 10
    rtc_idle_sleep_s: float = 0.01
    mocap_cfg: MocapConfig = field(
        default_factory=lambda: MocapConfig(domain_id=1, topic_name="MocapUE5G115Topic", depth=4)
    )
    body_pose_cfg: BodyPoseConfig = field(
        default_factory=lambda: BodyPoseConfig(domain_id=1, topic_name="WR/BodyPose_WR", depth=4)
    )
    camera_config: Path = Path("config/camera.yaml")


def extract_mocap_xyz_and_wxyz(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    expected_shape = (MOCAP_NUM_JOINTS, MOCAP_POS_DIM + MOCAP_QUAT_DIM)
    frame = np.asarray(frame, dtype=np.float32)
    if frame.shape != expected_shape:
        raise ValueError(f"Expected mocap frame shape {expected_shape}, got {frame.shape}")

    xyz = frame[:, :MOCAP_POS_DIM]
    wxyz = normalize_quaternion(frame[:, MOCAP_POS_DIM:])
    return xyz, wxyz


class StateHistoryQueue:
    def __init__(self, maxlen: int = 50):
        self._lock = threading.Lock()
        self._queue = deque(maxlen=maxlen)

    def put(self, msg: WR_GAE_BodyPose_Msg) -> None:
        body_joint = np.asarray(msg.q, dtype=np.float32).reshape(BODY_POSE_Q_SIZE).copy()
        imu = np.asarray(msg.wxyz, dtype=np.float32).reshape(BODY_POSE_WXYZ_SIZE).copy()
        with self._lock:
            self._queue.append(np.concatenate([imu, body_joint], axis=0))

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


def model_action_to_abs_action(
    action_output: np.ndarray,
    init_pose: np.ndarray,
    root_rel_cum: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    action_output = np.asarray(action_output, dtype=np.float32)
    action_with_root_delta = action_output.reshape(-1, 102)
    root_delta = action_with_root_delta[:, :3]
    action_without_root_delta = action_with_root_delta[:, 3:]

    action_mocap_xyz = action_without_root_delta[:, :33].reshape(-1, 11, 3)
    action_mocap_6d = action_without_root_delta[:, 33:].reshape(-1, 11, 6)
    action_mocap = np.concatenate([action_mocap_xyz, action_mocap_6d], axis=-1).reshape(-1, 99)
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
    return abs_action, next_root_rel_cum


def wait_for_root_pose(body_pose: BodyPoseSubscriber, poll_interval_s: float = 0.01) -> np.ndarray:
    while True:
        root_pose = body_pose.get_root_pose()
        if root_pose is not None:
            root_pose = root_pose.copy()
            root_pose[2] = 1.0
            return root_pose
        sleep(poll_interval_s)


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
        self.config = config
        self.client = server_client.PolicyClient(
            host=config.host,
            port=config.port,
            timeout_ms=config.timeout_ms,
        )
        self.body_pose = BodyPoseSubscriber(config.body_pose_cfg)
        self.publisher = MocapUE5G115MsgPublisher(config.mocap_cfg)
        self.state_history = StateHistoryQueue(maxlen=config.history_len)

        camera_name_list = get_camera_name(config.camera_config)
        self.camera_caps = {name: VideoCapture(name) for name in camera_name_list}

        self.root_pose: np.ndarray | None = None
        self.root_rel_cum: np.ndarray | None = None
        self.previous_actions: np.ndarray | None = None
        self.previous_actions_is_native: np.ndarray | None = None
        self.last_action: np.ndarray | None = None
        self.frame_counter = 1
        self.update_error: BaseException | None = None
        self.actions_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.update_actions_thread = threading.Thread(
            target=self.update_actions_rtc,
            daemon=True,
        )
        self.delay_buffer = DelayBuffer(
            initial_delay_frames=config.rtc_init_delay_frames,
            maxlen=config.rtc_delay_buffer_len,
        )

    def ping_server(self) -> None:
        print("Waiting for gr00t server to ping")
        if self.client.ping():
            print("Server is alive!")
            return
        print("Failed to connect to the server.")
        sys.exit(1)

    def start(self) -> None:
        self.ping_server()
        self.client.reset()
        self.root_pose = wait_for_root_pose(self.body_pose)
        self.state_history.put(wait_for_body_pose_msg(self.body_pose))
        self.update_actions_thread.start()

    def stop(self) -> None:
        self.stop_event.set()
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

    def build_observation(self) -> dict:
        state_history = self.state_history.get_all()
        delay_frames = self.delay_buffer.max()
        return build_observation_from_msg_rtc(
            state_history,
            self.config.task_description,
            frame=self.capture_frames(),
            history_len=self.config.history_len,
            delay_frames=delay_frames,
            action_executed_steps=self.config.rtc_action_exec_s,
        )

    def get_actions(self) -> tuple[np.ndarray, np.ndarray, int]:
        start_frame = self._current_frame_counter()
        observation = self.build_observation()
        response = self.client.get_action(observation)
        action_chunk_rel = np.asarray(response[0]["mocap"][0], dtype=np.float32).reshape(-1, 102)
        end_frame = self._current_frame_counter()
        measured_delay_frames = max(end_frame - start_frame, 0)
        delay_action_frames = max(
            self.config.rtc_init_delay_frames,
            self._native_action_index(measured_delay_frames),
        )
        action_chunk_rel = self._crop_delayed_actions(action_chunk_rel, delay_action_frames)

        if self.root_pose is None:
            raise RuntimeError("root_pose is not initialized")

        action_chunk_abs, self.root_rel_cum = model_action_to_abs_action(
            action_chunk_rel,
            self.root_pose,
            self.root_rel_cum,
        )
        if self.last_action is not None:
            action_chunk_abs = np.concatenate((self.last_action, action_chunk_abs), axis=0)

        action_chunk_abs, action_is_native = self._interpolate_with_native_mask(action_chunk_abs)

        self.last_action = action_chunk_abs[-1:].copy()
        return action_chunk_abs, action_is_native, delay_action_frames

    def update_actions_rtc(self) -> None:
        try:
            actions, actions_is_native, delay_frames = self.get_actions()
        except Exception as exc:
            self.update_error = exc
            self.stop_event.set()
            logger.exception("Initial RTC action update failed.")
            return

        self.delay_buffer.add(delay_frames)
        with self.actions_lock:
            self.previous_actions = actions.copy()
            self.previous_actions_is_native = actions_is_native.copy()
            self.frame_counter = 1

        while not self.stop_event.is_set():
            should_update = False
            with self.actions_lock:
                if self.previous_actions is None:
                    should_update = True
                elif self.frame_counter > self.config.rtc_action_exec_s:
                    should_update = True

            if should_update:
                try:
                    actions, actions_is_native, delay_frames = self.get_actions()
                except Exception:
                    logger.exception("RTC action update failed; retrying.")
                    sleep(0.5)
                    continue

                self.delay_buffer.add(delay_frames)
                with self.actions_lock:
                    self.previous_actions = actions.copy()
                    self.previous_actions_is_native = actions_is_native.copy()
                    self.frame_counter = 1

            sleep(self.config.rtc_idle_sleep_s)

    def action_execution_rtc(self, roll_out_len: int) -> None:
        rate = RateLimiter(frequency=self.config.send_fps)
        init_frame = SIM_INIT_POSE.astype(np.float32)
        frames_sent = 0

        while frames_sent < roll_out_len:
            if self.update_error is not None:
                raise RuntimeError("RTC update thread failed") from self.update_error
            if self.stop_event.is_set():
                break

            with self.actions_lock:
                actions = self.previous_actions
                actions_is_native = self.previous_actions_is_native
                is_native_action = False
                if actions is None or actions.shape[0] == 0:
                    mocap_frame = init_frame
                else:
                    raw_action_index = self.frame_counter - 1
                    action_index = min(raw_action_index, actions.shape[0] - 1)
                    mocap_frame = actions[action_index]
                    if actions_is_native is not None and raw_action_index < actions.shape[0]:
                        is_native_action = bool(actions_is_native[action_index])
                    self.frame_counter += 1

            xyz, wxyz = extract_mocap_xyz_and_wxyz(mocap_frame)
            self.publisher.send_msg(fps=self.config.send_fps, xyz=xyz, wxyz=wxyz)
            if is_native_action:
                self.put_current_body_pose_to_history()
            frames_sent += 1
            rate.sleep()

    def put_current_body_pose_to_history(self) -> None:
        msg = self.body_pose.get_msg()
        if msg is None:
            logger.warning("Skip state history update: body pose message is unavailable.")
            return
        self.state_history.put(msg)

    def _interpolate_with_native_mask(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if (
            not self.config.use_interpolate
            or self.config.num_interp <= 0
            or actions.shape[0] <= 1
        ):
            return actions, np.ones(actions.shape[0], dtype=bool)

        stride = self.config.num_interp + 1
        num_interp_frames = actions.shape[0] + (actions.shape[0] - 1) * self.config.num_interp
        native_mask = np.zeros(num_interp_frames, dtype=bool)
        native_mask[::stride] = True
        return interpolate_pose7(actions, self.config.num_interp), native_mask

    @staticmethod
    def _crop_delayed_actions(actions: np.ndarray, delay_frames: int) -> np.ndarray:
        if actions.shape[0] == 0:
            return actions
        delay_frames = min(max(delay_frames, 0), actions.shape[0] - 1)
        return actions[delay_frames:].copy()

    def _current_frame_counter(self) -> int:
        with self.actions_lock:
            return self.frame_counter

    def _native_action_index(self, send_frame_index: int) -> int:
        if not self.config.use_interpolate:
            return max(int(send_frame_index), 0)
        return max(int(send_frame_index), 0) // (self.config.num_interp + 1)

def main() -> None:
    config = tyro.cli(ClientConfig)
    runner = G1Gr00tRtcRunner(config)
    try:
        runner.start()
        runner.action_execution_rtc(config.roll_out)
    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    finally:
        runner.stop()
        logger.info("RTC inference stopped.")


if __name__ == "__main__":
    main()

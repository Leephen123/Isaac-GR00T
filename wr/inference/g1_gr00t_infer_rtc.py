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
    SELECT_11_INDICES,
    build_observation_from_msg_with_history,
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
    history_len: int = 50
    roll_out: int = 5000
    rtc_action_exec_s: int = 25
    rtc_init_delay_frames: int = 15
    rtc_delay_buffer_len: int = 3
    rtc_idle_sleep_s: float = 0.01
    rtc_beta: float = 5.0
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


class StateHistoryQueue:
    def __init__(self, maxlen: int = 50):
        self._lock = threading.Lock()
        self._queue = deque(maxlen=maxlen)
        self._last_timestamp: int | None = None

    def put(self, msg: WR_GAE_BodyPose_Msg) -> bool:
        state = msg_to_6d_rot_joint(msg)
        timestamp = int(msg.timestamp)
        with self._lock:
            if timestamp == self._last_timestamp:
                return False
            self._queue.append(state)
            self._last_timestamp = timestamp
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


def model_action_to_abs_action(
    action_output: np.ndarray,
    init_pose: np.ndarray,
    root_rel_cum: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
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


def wait_for_root_pose(
    body_pose: BodyPoseSubscriber, poll_interval_s: float = 0.01
) -> np.ndarray:
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
        if config.send_fps <= 0:
            raise ValueError(f"send_fps must be positive, got {config.send_fps}")
        if config.history_len <= 0:
            raise ValueError(f"history_len must be positive, got {config.history_len}")
        if config.rtc_action_exec_s <= 0:
            raise ValueError(
                f"rtc_action_exec_s must be positive, got {config.rtc_action_exec_s}"
            )
        if config.rtc_idle_sleep_s <= 0:
            raise ValueError(
                f"rtc_idle_sleep_s must be positive, got {config.rtc_idle_sleep_s}"
            )
        if not np.isfinite(config.rtc_beta) or config.rtc_beta <= 0:
            raise ValueError(f"rtc_beta must be positive, got {config.rtc_beta}")

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
        self.previous_inference_start_step: int | None = None
        self.state_horizon: int | None = config.history_len
        self.global_exec_step = 0
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
            self.state_horizon = self._server_state_horizon()
            logger.info("Server state horizon: %s", self.state_horizon)
            self.client_ready_event.set()
            return
        raise ConnectionError("Failed to connect to the gr00t server")

    def start(self) -> None:
        self.root_pose = wait_for_root_pose(self.body_pose)
        self.state_history.put(wait_for_body_pose_msg(self.body_pose))
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
        return build_observation_from_msg_with_history(
            state_history,
            self.config.task_description,
            frame=frames,
            history_len=self.config.history_len,
            state_horizon=self.state_horizon,
        )

    def get_actions(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
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
        ).reshape(-1, 102)

        if self.root_pose is None:
            raise RuntimeError("root_pose is not initialized")

        action_chunk_abs, _ = model_action_to_abs_action(
            action_chunk_rel,
            self.root_pose,
            chunk_start_root_rel,
        )
        measured_delay_frames = max(
            self._current_global_exec_step() - inference_start_step, 0
        )
        if measured_delay_frames >= action_chunk_rel.shape[0]:
            raise RuntimeError(
                "Discard stale action chunk: "
                f"delay={measured_delay_frames}, horizon={action_chunk_rel.shape[0]}"
            )

        return (
            action_chunk_abs,
            action_chunk_rel,
            chunk_start_root_rel,
            inference_start_step,
            measured_delay_frames,
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
                        action_chunk_rel,
                        chunk_start_root_rel,
                        inference_start_step,
                        delay_frames,
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

                with self.actions_lock:
                    if self.previous_inference_start_step is not None:
                        self.delay_buffer.add(delay_frames)
                    self.previous_actions = actions.copy()
                    self.previous_action_chunk_rel = action_chunk_rel.copy()
                    self.previous_chunk_start_root_rel = chunk_start_root_rel.copy()
                    self.previous_inference_start_step = inference_start_step
                    self.force_normal_next = False

            sleep(self.config.rtc_idle_sleep_s)

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
                if actions is None or actions.shape[0] == 0:
                    sent_action = False
                else:
                    if self.previous_inference_start_step is None:
                        raise RuntimeError(
                            "Action chunk is missing its inference start step"
                        )
                    raw_action_index = (
                        self.global_exec_step - self.previous_inference_start_step
                    )
                    action_index = max(0, min(raw_action_index, actions.shape[0] - 1))
                    mocap_frame = actions[action_index]
                    xyz, wxyz = extract_mocap_xyz_and_wxyz(mocap_frame)
                    self.publisher.send_msg(
                        fps=self.config.send_fps,
                        xyz=xyz,
                        wxyz=wxyz,
                    )
                    self.put_current_body_pose_to_history()
                    self.global_exec_step += 1
                    sent_action = True

            if not sent_action:
                rate.sleep()
                continue

            frames_sent += 1
            rate.sleep()

    def put_current_body_pose_to_history(self) -> None:
        msg = self.body_pose.get_msg()
        if msg is None:
            logger.warning(
                "Skip state history update: body pose message is unavailable."
            )
            return
        self.state_history.put(msg)

    def _current_global_exec_step(self) -> int:
        with self.actions_lock:
            return self.global_exec_step

    def _server_state_horizon(self) -> int:
        if self.client is None or not hasattr(self.client, "get_modality_config"):
            return self.config.history_len
        try:
            modality_config = self.client.get_modality_config()
            state_config = modality_config.get("state")
            delta_indices = (
                state_config.get("delta_indices")
                if isinstance(state_config, dict)
                else getattr(state_config, "delta_indices", None)
            )
            if delta_indices:
                return len(delta_indices)
        except Exception:
            logger.exception(
                "Failed to read server modality config; using history_len=%d",
                self.config.history_len,
            )
        return self.config.history_len

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
            or previous_action_chunk_rel.shape[1] != 102
            or previous_action_chunk_rel.shape[0] == 0
        ):
            raise ValueError(
                "Expected previous decoded actions with shape (H, 102), "
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

        # Delta[t] moves frame t to t + 1. Consume one delta for every elapsed
        # execution step; consuming all H deltas gives the next chunk anchor.
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

    @staticmethod
    def _normalize_model_actions(actions: np.ndarray) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 3:
            raise ValueError(
                f"Expected normalized model actions with shape (B, H, D), got {actions.shape}"
            )
        if actions.shape[0] != 1 or actions.shape[1] == 0 or actions.shape[2] == 0:
            raise ValueError(f"Invalid normalized model action shape: {actions.shape}")
        return actions.copy()


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

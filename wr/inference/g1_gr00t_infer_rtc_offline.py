import json
import sys
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from time import sleep

import numpy as np
import tyro
from loop_rate_limiters import RateLimiter
from PIL import Image

sys.path.insert(0, "/liujinxin/liyifan/wr/")
from data_res.dds import BODY_POSE_Q_SIZE, BODY_POSE_WXYZ_SIZE
from data_res.log import get_logger
from data_res.transforms import (
    compute_absolute,
    compute_imu_relative,
    normalize_quaternion,
    quaternion_to_rotation_6d,
    restore_mocap_from_root_relative,
    rotation_6d_to_quaternion,
)
from data_res.utils import SELECT_11_INDICES, format_history_state_for_observation
from inference.utils import SIM_INIT_POSE
from serve_res.gr00t import server_client

logger = get_logger(__name__)

DEFAULT_TASK_DESCRIPTION = (
    "Clean the items on the desk and put the doll on the left sofa"
)

CAMERA_KEY_ALIASES = {
    "front_head": ("front_head", "front", "ego_view", "scene"),
    "left_hand": ("left_hand", "left", "left_wrist_view", "left_wrist"),
    "right_hand": ("right_hand", "right", "right_wrist_view", "right_wrist"),
}

VIDEO_KEY_MAP = {
    "front_head": "ego_view",
    "left_hand": "left_wrist_view",
    "right_hand": "right_wrist_view",
}


@dataclass
class ClientConfig:
    replay_data_path: Path = Path(
        "/liujinxin/liyifan/wr/data/episode_0/data_root_relative_6D.json"
    )
    host: str = "127.0.0.1"
    port: int = 9002
    timeout_ms: int = 15000
    task_description: str = DEFAULT_TASK_DESCRIPTION
    send_fps: float = 50.0
    history_len: int = 50
    roll_out: int = 5000
    loop_data: bool = True
    rtc_action_exec_s: int = 10
    rtc_init_delay_frames: int = 2
    rtc_delay_buffer_len: int = 10
    rtc_idle_sleep_s: float = 0.01
    root_pose_z: float | None = 1.0
    save_executed_actions_path: Path | None = None
    log_interval_frames: int = 50


def extract_mocap_xyz_and_wxyz(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    expected_shape = (15, 7)
    frame = np.asarray(frame, dtype=np.float32)
    if frame.shape != expected_shape:
        raise ValueError(
            f"Expected mocap frame shape {expected_shape}, got {frame.shape}"
        )

    xyz = frame[:, :3]
    wxyz = normalize_quaternion(frame[:, 3:7])
    return xyz, wxyz


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
    action_mocap = np.concatenate([action_mocap_xyz, action_mocap_6d], axis=-1).reshape(
        -1, 99
    )
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


def load_image_from_path_or_array(
    image_input: Path | np.ndarray,
    target_size: tuple[int, int] = (256, 256),
) -> np.ndarray:
    if isinstance(image_input, Path):
        image = Image.open(image_input).convert("RGB")
    else:
        image_array = np.asarray(image_input)
        if image_array.ndim != 3 or image_array.shape[-1] != 3:
            raise ValueError(f"Expected image shape (H, W, 3), got {image_array.shape}")
        image_array = image_array.astype(np.uint8)
        image = Image.fromarray(image_array[:, :, ::-1]).convert("RGB")

    image = image.resize(target_size, Image.BILINEAR)
    return np.asarray(image, dtype=np.uint8)[None, None, ...]


def history_state_to_imu_joints(
    history_state: np.ndarray, history_len: int
) -> np.ndarray:
    history_state = np.asarray(history_state, dtype=np.float32)
    raw_dim = BODY_POSE_WXYZ_SIZE + BODY_POSE_Q_SIZE
    rot6d_dim = 6 + BODY_POSE_Q_SIZE
    if history_state.ndim != 2 or history_state.shape[1] not in (raw_dim, rot6d_dim):
        raise ValueError(
            f"history_state shape error: expected (T, {raw_dim}) or (T, {rot6d_dim}), "
            f"got {history_state.shape}"
        )

    if history_state.shape[0] > history_len:
        history_state = history_state[-history_len:]
    elif 0 < history_state.shape[0] < history_len:
        pad_count = history_len - history_state.shape[0]
        history_state = np.concatenate(
            [np.repeat(history_state[:1], pad_count, axis=0), history_state],
            axis=0,
        )

    if history_state.shape[1] == rot6d_dim:
        return history_state.astype(np.float32, copy=False)

    imu = history_state[:, :BODY_POSE_WXYZ_SIZE]
    body_joint = history_state[:, BODY_POSE_WXYZ_SIZE:]
    imu = compute_imu_relative(imu, imu)
    imu_6d = quaternion_to_rotation_6d(imu)
    return np.concatenate([imu_6d, body_joint], axis=1).astype(np.float32)


def build_observation_from_replay(
    history_state: np.ndarray,
    task_description: str,
    frame: dict[str, np.ndarray],
    history_len: int,
    state_horizon: int | None = None,
) -> dict:
    state = history_state_to_imu_joints(history_state, history_len=history_len)
    video = {
        "ego_view": np.zeros((1, 1, 256, 256, 3), dtype=np.uint8),
        "left_wrist_view": np.zeros((1, 1, 256, 256, 3), dtype=np.uint8),
        "right_wrist_view": np.zeros((1, 1, 256, 256, 3), dtype=np.uint8),
    }

    for camera_name, image in frame.items():
        if camera_name not in VIDEO_KEY_MAP:
            logger.warning(
                "Ignoring unknown camera key in replay frame: %s", camera_name
            )
            continue
        video[VIDEO_KEY_MAP[camera_name]] = load_image_from_path_or_array(image)

    observation = {
        "video": video,
        "state": {
            "imu_joints": format_history_state_for_observation(
                state, state_horizon=state_horizon
            )
        },
        "language": {"annotation.human.task_description": [[task_description]]},
        "stickman": {
            "annotation.human.stickman": np.zeros((1, 1, 900), dtype=np.float32)
        },
    }
    return observation


class StateHistoryQueue:
    def __init__(self, maxlen: int):
        self._lock = threading.Lock()
        self._queue = deque(maxlen=maxlen)

    def put_state(self, state: np.ndarray) -> None:
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        valid_dims = (BODY_POSE_WXYZ_SIZE + BODY_POSE_Q_SIZE, 6 + BODY_POSE_Q_SIZE)
        if state.shape[0] not in valid_dims:
            raise ValueError(f"Expected state dim in {valid_dims}, got {state.shape}")
        with self._lock:
            self._queue.append(state.copy())

    def get_all(self) -> np.ndarray:
        with self._lock:
            if not self._queue:
                raise RuntimeError("state history is empty")
            states = [state.copy() for state in self._queue]

        maxlen = self._queue.maxlen
        if len(states) < maxlen:
            states = [states[0].copy()] * (maxlen - len(states)) + states
        return np.stack(states, axis=0)


class DelayBuffer:
    def __init__(self, initial_delay_frames: int, maxlen: int):
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


class OfflineReplay:
    def __init__(self, data_path: Path, loop_data: bool):
        self.data_path = data_path
        self.data_root = data_path.parent
        self.loop_data = loop_data
        self.records = self._load_records(data_path)
        self._lock = threading.Lock()
        self._index = 0

    @staticmethod
    def _load_records(data_path: Path) -> list[dict]:
        with data_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            for key in ("data", "records", "steps"):
                if key in data:
                    data = data[key]
                    break

        if not isinstance(data, list) or len(data) == 0:
            raise ValueError(
                f"Expected non-empty list-like replay json, got {type(data)}"
            )
        return data

    @property
    def index(self) -> int:
        with self._lock:
            return self._index

    def current_state(self) -> np.ndarray:
        with self._lock:
            return self._record_to_state(self.records[self._index])

    def current_frames(self) -> dict[str, np.ndarray]:
        with self._lock:
            record = self.records[self._index]
        return self._record_to_frames(record)

    def advance_state(self) -> np.ndarray | None:
        with self._lock:
            if self._index + 1 < len(self.records):
                self._index += 1
            elif self.loop_data:
                self._index = 0
            else:
                return None
            return self._record_to_state(self.records[self._index])

    def root_pose(self, z_override: float | None = None) -> np.ndarray:
        with self._lock:
            record = self.records[self._index]
        if "root_pose" in record:
            root_pose = np.asarray(record["root_pose"], dtype=np.float32).reshape(7)
        elif "mocap" in record:
            mocap = np.asarray(record["mocap"], dtype=np.float32)
            if mocap.shape == (15, 7):
                root_pose = mocap[0].copy()
            else:
                root_pose = SIM_INIT_POSE[0].astype(np.float32).copy()
        else:
            root_pose = SIM_INIT_POSE[0].astype(np.float32).copy()

        if z_override is not None:
            root_pose[2] = float(z_override)
        return root_pose

    def task_description(self) -> str | None:
        task = self.records[0].get("task")
        if isinstance(task, list) and task:
            return str(task[0])
        if isinstance(task, str):
            return task
        return None

    def _record_to_state(self, record: dict) -> np.ndarray:
        if "state" in record:
            state = np.asarray(record["state"], dtype=np.float32).reshape(-1)
            if state.shape[0] in (
                BODY_POSE_WXYZ_SIZE + BODY_POSE_Q_SIZE,
                6 + BODY_POSE_Q_SIZE,
            ):
                return state

        imu = self._read_first_existing(record, ("imu", "wxyz", "base_wxyz"))
        body_joint = self._read_first_existing(record, ("body_joint", "q", "joint_q"))
        if imu is None or body_joint is None:
            raise KeyError(
                "Replay record must contain raw imu/wxyz and body_joint/q fields"
            )

        imu = np.asarray(imu, dtype=np.float32).reshape(-1)
        body_joint = np.asarray(body_joint, dtype=np.float32).reshape(-1)
        if imu.shape[0] not in (BODY_POSE_WXYZ_SIZE, 6):
            raise ValueError(
                f"Expected imu shape (4,) raw wxyz or (6,) rotation-6d, got {imu.shape}"
            )
        if body_joint.shape[0] < BODY_POSE_Q_SIZE:
            raise ValueError(
                f"Expected at least {BODY_POSE_Q_SIZE} body joints, got {body_joint.shape}"
            )
        return np.concatenate([imu, body_joint[:BODY_POSE_Q_SIZE]], axis=0).astype(
            np.float32
        )

    def _record_to_frames(self, record: dict) -> dict[str, np.ndarray]:
        frames = {}
        for camera_name, aliases in CAMERA_KEY_ALIASES.items():
            image_path = self._read_first_existing(record, aliases)
            if image_path is None:
                continue
            frames[camera_name] = self._load_image_as_bgr(image_path)
        return frames

    def _load_image_as_bgr(self, image_path: str | Path) -> np.ndarray:
        path = Path(image_path)
        if not path.is_absolute():
            path = self.data_root / path
        with Image.open(path).convert("RGB") as image:
            rgb = np.asarray(image, dtype=np.uint8)
        return rgb[:, :, ::-1].copy()

    @staticmethod
    def _read_first_existing(record: dict, keys: tuple[str, ...]):
        for key in keys:
            if key in record:
                return record[key]
        return None


class G1Gr00tRtcOfflineRunner:
    def __init__(self, config: ClientConfig):
        self.config = config
        if config.rtc_action_exec_s <= 0:
            raise ValueError("rtc_action_exec_s must be positive")
        self.replay = OfflineReplay(config.replay_data_path, loop_data=config.loop_data)
        self.client = server_client.PolicyClient(
            host=config.host,
            port=config.port,
            timeout_ms=config.timeout_ms,
        )
        self.state_history = StateHistoryQueue(maxlen=config.history_len)
        self.delay_buffer = DelayBuffer(
            initial_delay_frames=config.rtc_init_delay_frames,
            maxlen=config.rtc_delay_buffer_len,
        )

        self.previous_actions: np.ndarray | None = None
        self.previous_normalized_actions: np.ndarray | None = None
        self.previous_inference_start_step: int | None = None
        self.state_horizon: int | None = config.history_len
        self.global_exec_step = 0
        self.update_error: BaseException | None = None
        self.actions_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.update_actions_thread = threading.Thread(
            target=self.update_actions_rtc,
            daemon=True,
        )
        self.executed_actions: list[np.ndarray] = []

        task_from_data = self.replay.task_description()
        if task_from_data and config.task_description == DEFAULT_TASK_DESCRIPTION:
            self.config.task_description = task_from_data

    def ping_server(self) -> None:
        print("Waiting for gr00t server to ping")
        if self.client.ping():
            print("Server is alive!")
            self.state_horizon = self._server_state_horizon()
            logger.info("Server state horizon: %s", self.state_horizon)
            return
        print("Failed to connect to the server.")
        sys.exit(1)

    def start(self) -> None:
        self.ping_server()
        self.client.reset()
        self.state_history.put_state(self.replay.current_state())
        self.update_actions_thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.update_actions_thread.is_alive():
            self.update_actions_thread.join(timeout=2.0)
        self.save_executed_actions()

    def build_observation(self) -> dict:
        return build_observation_from_replay(
            self.state_history.get_all(),
            self.config.task_description,
            frame=self.replay.current_frames(),
            history_len=self.config.history_len,
            state_horizon=self.state_horizon,
        )

    def get_actions(self) -> tuple[np.ndarray, np.ndarray, int, int]:
        with self.actions_lock:
            inference_start_step = self.global_exec_step
            previous_normalized_actions = (
                None
                if self.previous_normalized_actions is None
                else self.previous_normalized_actions.copy()
            )
            previous_inference_start_step = self.previous_inference_start_step

        root_pose = self.replay.root_pose(z_override=self.config.root_pose_z)
        observation = self.build_observation()
        rtc_inputs = self._build_rtc_inputs(
            inference_start_step,
            previous_inference_start_step,
            previous_normalized_actions,
        )
        if rtc_inputs is not None:
            observation.update(rtc_inputs)

        response = self.client.get_action(observation)
        action_chunk_rel = np.asarray(
            response[0]["mocap"][0], dtype=np.float32
        ).reshape(-1, 102)
        normalized_actions = self._normalize_model_actions(response[1])
        if normalized_actions.shape[1] != action_chunk_rel.shape[0]:
            raise ValueError(
                "Decoded and normalized action horizons differ: "
                f"{action_chunk_rel.shape[0]} vs {normalized_actions.shape[1]}"
            )

        action_chunk_abs, _ = model_action_to_abs_action(
            action_chunk_rel,
            root_pose,
            root_rel_cum=root_pose[:3],
        )
        delay_frames = max(
            self._current_global_exec_step() - inference_start_step,
            0,
        )
        if delay_frames >= action_chunk_abs.shape[0]:
            raise RuntimeError(
                "Inference delay reached or exceeded the action horizon: "
                f"d={delay_frames}, H={action_chunk_abs.shape[0]}"
            )
        return (
            action_chunk_abs,
            normalized_actions,
            inference_start_step,
            delay_frames,
        )

    def update_actions_rtc(self) -> None:
        try:
            (
                actions,
                normalized_actions,
                inference_start_step,
                delay_frames,
            ) = self.get_actions()
        except Exception as exc:
            self.update_error = exc
            self.stop_event.set()
            logger.exception("Initial offline RTC action update failed.")
            return

        self.delay_buffer.add(delay_frames)
        with self.actions_lock:
            self.previous_actions = actions.copy()
            self.previous_normalized_actions = normalized_actions
            self.previous_inference_start_step = inference_start_step

        while not self.stop_event.is_set():
            with self.actions_lock:
                steps_since_inference = (
                    self.global_exec_step - self.previous_inference_start_step
                )
                should_update = steps_since_inference >= self.config.rtc_action_exec_s

            if should_update:
                try:
                    (
                        actions,
                        normalized_actions,
                        inference_start_step,
                        delay_frames,
                    ) = self.get_actions()
                except Exception:
                    logger.exception("Offline RTC action update failed; retrying.")
                    sleep(0.5)
                    continue

                self.delay_buffer.add(delay_frames)
                with self.actions_lock:
                    self.previous_actions = actions.copy()
                    self.previous_normalized_actions = normalized_actions
                    self.previous_inference_start_step = inference_start_step

            sleep(self.config.rtc_idle_sleep_s)

    def action_execution_rtc(self, roll_out_len: int) -> None:
        rate = RateLimiter(frequency=self.config.send_fps)
        frames_sent = 0
        logged_waiting_for_initial_actions = False

        while frames_sent < roll_out_len:
            if self.update_error is not None:
                raise RuntimeError(
                    "Offline RTC update thread failed"
                ) from self.update_error
            if self.stop_event.is_set():
                break

            with self.actions_lock:
                actions = self.previous_actions
                if actions is None or actions.shape[0] == 0:
                    mocap_frame = None
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
                    self.global_exec_step += 1

            if mocap_frame is None:
                if not logged_waiting_for_initial_actions:
                    logger.info("Waiting for initial offline RTC action chunk.")
                    logged_waiting_for_initial_actions = True
                sleep(self.config.rtc_idle_sleep_s)
                continue
            logged_waiting_for_initial_actions = False

            # Validate the generated action shape and quaternion normalization, but do not publish it.
            extract_mocap_xyz_and_wxyz(mocap_frame)
            self.executed_actions.append(
                np.asarray(mocap_frame, dtype=np.float32).copy()
            )

            if not self.advance_replay_state():
                logger.info("Replay data exhausted; stopping offline execution.")
                break

            frames_sent += 1
            if (
                self.config.log_interval_frames > 0
                and frames_sent % self.config.log_interval_frames == 0
            ):
                logger.info(
                    "offline frame=%d replay_index=%d action_buffer=%s",
                    frames_sent,
                    self.replay.index,
                    None if actions is None else actions.shape,
                )
            rate.sleep()

    def advance_replay_state(self) -> bool:
        state = self.replay.advance_state()
        if state is None:
            return False
        self.state_history.put_state(state)
        return True

    def save_executed_actions(self) -> None:
        if self.config.save_executed_actions_path is None or not self.executed_actions:
            return
        path = self.config.save_executed_actions_path
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, np.stack(self.executed_actions, axis=0))
        logger.info("Saved offline executed actions: %s", path)

    def _current_global_exec_step(self) -> int:
        with self.actions_lock:
            return self.global_exec_step

    def _server_state_horizon(self) -> int:
        if not hasattr(self.client, "get_modality_config"):
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

    def _build_rtc_inputs(
        self,
        inference_start_step: int,
        previous_inference_start_step: int | None,
        previous_normalized_actions: np.ndarray | None,
    ) -> dict | None:
        if previous_inference_start_step is None or previous_normalized_actions is None:
            return None

        horizon = previous_normalized_actions.shape[1]
        action_executed_steps = inference_start_step - previous_inference_start_step
        delay_frames = self.delay_buffer.max()
        if not delay_frames <= action_executed_steps <= horizon - delay_frames:
            logger.warning(
                "Skipping RTC guidance because d <= s <= H - d is not satisfied: "
                "d=%s, s=%s, H=%s",
                delay_frames,
                action_executed_steps,
                horizon,
            )
            return None

        return {
            "delay_frames": np.asarray(delay_frames, dtype=np.int32),
            "action_executed_steps": np.asarray(action_executed_steps, dtype=np.int32),
        }

    @staticmethod
    def _normalize_model_actions(response_actions) -> np.ndarray:
        actions = np.asarray(response_actions, dtype=np.float32)
        if actions.ndim != 3:
            raise ValueError(
                "Expected normalized model actions with shape (B, H, D), "
                f"got {actions.shape}"
            )
        if actions.shape[0] != 1 or actions.shape[1] == 0 or actions.shape[2] == 0:
            raise ValueError(f"Invalid normalized model action shape: {actions.shape}")
        return actions.copy()


def main() -> None:
    config = tyro.cli(ClientConfig)
    runner = G1Gr00tRtcOfflineRunner(config)
    try:
        runner.start()
        runner.action_execution_rtc(config.roll_out)
    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    finally:
        runner.stop()
        logger.info("Offline RTC inference stopped.")


if __name__ == "__main__":
    main()

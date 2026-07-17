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
    interpolate_pose7,
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
    task_description: str = (
        "Clean the items on the desk and put the doll on the left sofa"
    )
    send_fps: float = 50.0
    history_len: int = 50
    roll_out: int = 5000
    action_exec_s: int = 25
    idle_sleep_s: float = 0.01
    use_chunk_interpolate: bool = False
    chunk_interp_frames: int = 3
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


def interpolate_chunk_transition(
    previous_action: np.ndarray,
    next_action: np.ndarray,
    num_interp: int,
) -> np.ndarray:
    """Return only the inserted transition frames between two actions."""
    if num_interp <= 0:
        return np.zeros((0, MOCAP_NUM_JOINTS, 7), dtype=np.float32)

    action_pair = np.stack((previous_action, next_action), axis=0)
    return interpolate_pose7(action_pair, num_interp)[1:-1]


class StateHistoryQueue:
    def __init__(self, maxlen: int = 50):
        self._lock = threading.Lock()
        self._queue = deque(maxlen=maxlen)

    def put(self, msg: WR_GAE_BodyPose_Msg) -> None:
        with self._lock:
            self._queue.append(msg_to_6d_rot_joint(msg))

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

        image = image.astype(np.uint8)

        image = image[:, :, ::-1]
        image = Image.fromarray(image).convert("RGB")

    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    if debug:
        os.makedirs("./image_debug", exist_ok=True)
        save_name = name if name is not None else "debug_image"
        save_path = os.path.join("./image_debug", f"{save_name}.png")
        image.save(save_path)
        print(f"image save in {save_path}")

    resized_image = image.resize(target_size, Image.BILINEAR)
    result = np.array(resized_image)

    return result[None, None, ...]


def build_observation_from_msg_with_history(
    history_state,
    task_description: str,
    frame=None,
):
    history_state = np.asarray(history_state, dtype=np.float32)

    video = {
        "ego_view": np.zeros((1, 1, 256, 256, 3), dtype=np.uint8),
        "left_wrist_view": np.zeros((1, 1, 256, 256, 3), dtype=np.uint8),
        "right_wrist_view": np.zeros((1, 1, 256, 256, 3), dtype=np.uint8),
    }

    if frame is not None:
        for name, image in frame.items():
            video[CAMERAS_MAP[name]] = load_image_from_path_or_array(
                np.asarray(image), (256, 256)
            )
    state = history_state.reshape(-1)
    observation = {
        "video": video,
        "state": {"imu_joints": state[None, None, :].astype(np.float32)},
        "language": {"annotation.human.task_description": [[task_description]]},
    }

    # imu = history_state[:, :6]
    # left_leg = history_state[:, 6:12]
    # right_leg = history_state[:, 12:18]
    # waist = history_state[:, 18:21]
    # left_arm = history_state[:, 21:28]
    # right_arm = history_state[:, 28:35]

    # observation = {
    #    "video": video,
    #    "state": {"imu": imu[None, :].astype(np.float32),
    #              "left_leg": left_leg[None, :],
    #              "right_leg": right_leg[None, :],
    #              "waist": waist[None, :],
    #              "left_arm": left_arm[None, :],
    #              "right_arm": right_arm[None, :]},
    #    "language": {"annotation.human.task_description": [[task_description]]},
    # }

    stickman_np = np.zeros((1, 1, 900), dtype=np.float32)
    observation["stickman"] = {"annotation.human.stickman": stickman_np}
    return observation


def model_action_to_abs_action(
    action_output: np.ndarray,
    init_pose: np.ndarray,
    root_rel_cum: np.ndarray | None = None,
) -> np.ndarray:
    action_output = np.asarray(action_output, dtype=np.float32)
    logger.info("action output shape: %s", action_output.shape)

    action_with_root_delta = action_output.reshape(-1, 102)
    root_delta = action_with_root_delta[:, :3]
    action_without_root_delta = action_with_root_delta[:, 3:]

    action_mocap_xyz = action_without_root_delta[:, :33].reshape(-1, 11, 3)
    action_mocap_6d = action_without_root_delta[:, 33:].reshape(-1, 11, 6)
    action_mocap = np.concatenate(
        [action_mocap_xyz, action_mocap_6d],
        axis=-1,
    ).reshape(-1, 99)
    action_mocap = np.concatenate([root_delta, action_mocap], axis=-1)

    if root_rel_cum is None:
        action_11x9, _ = restore_mocap_from_root_relative(action_mocap)
    else:
        action_11x9, _ = restore_mocap_from_root_relative(
            action_mocap,
            root_rel_cum,
        )

    num_frames = action_11x9.shape[0]
    action_15x9 = np.zeros((num_frames, 15, 9), dtype=np.float32)
    action_15x9[..., 0:3] = 0.0
    action_15x9[..., 3:9] = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    action_15x9[:, SELECT_11_INDICES, :] = action_11x9

    action_15x7 = rotation_6d_to_quaternion(action_15x9)
    num_frames, num_joints, num_poses = action_15x7.shape
    action_15x7_flat = action_15x7.reshape(num_frames * num_joints, num_poses)
    init_pose_x7_flat = np.tile(init_pose, (num_frames * num_joints, 1))

    ans = compute_absolute(init_pose_x7_flat, action_15x7_flat).reshape(
        num_frames, num_joints, num_poses
    )
    return ans

def wait_for_body_pose_msg(
    body_pose: BodyPoseSubscriber,
    poll_interval_s: float = 0.01,
) -> WR_GAE_BodyPose_Msg:
    while True:
        msg = body_pose.get_msg()
        if msg is not None:
            return msg
        sleep(poll_interval_s)


class G1Gr00tAsyncRunner:
    def __init__(self, config: ClientConfig):
        if config.action_exec_s <= 0:
            raise ValueError(
                f"action_exec_s must be positive, got {config.action_exec_s}"
            )
        if config.chunk_interp_frames < 0:
            raise ValueError(
                "chunk_interp_frames must be non-negative, "
                f"got {config.chunk_interp_frames}"
            )

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
        self.previous_actions: np.ndarray | None = None
        self.previous_action_chunk_rel: np.ndarray | None = None
        self.previous_chunk_start_root_rel: np.ndarray | None = None
        self.previous_inference_start_step: int | None = None
        self.global_exec_step = 0
        self.update_error: BaseException | None = None
        self.actions_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.update_actions_thread = threading.Thread(
            target=self.update_actions_async,
            daemon=True,
        )

    def start(self) -> None:
        print("Waiting for gr00t server to ping")
        if not self.client.ping():
            print("Failed to connect to the server.")
            sys.exit(1)
        print("Server is alive!")

        initial_msg = wait_for_body_pose_msg(self.body_pose)
        self.root_pose = self.body_pose.msg_to_root_pose(initial_msg)
        self.root_pose[2] = 1.0
        self.state_history.put(initial_msg)
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
            for _ in range(5):
                ok, frame = camera_cap.read()
                if not ok or frame is None:
                    raise RuntimeError(f"Failed to read camera frame from {name}")
                frames[name] = frame
        return frames

    def get_actions(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
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
            state_history = self.state_history.get_all()

        chunk_start_root_rel = self._chunk_start_root_rel(
            inference_start_step,
            previous_inference_start_step,
            previous_action_chunk_rel,
            previous_chunk_start_root_rel,
        )
        observation = build_observation_from_msg_with_history(
            state_history,
            self.config.task_description,
            frame=frames,
            history_len=self.config.history_len,
        )
        response = self.client.get_action(observation)
        action_chunk_rel = np.asarray(
            response[0]["mocap"][0], dtype=np.float32
        )
        with self.actions_lock:
            inference_delay_steps = self.global_exec_step - inference_start_step
        if inference_delay_steps >= action_chunk_rel.shape[0]:
            raise RuntimeError(
                "Discard stale action chunk: "
                f"delay={inference_delay_steps}, horizon={action_chunk_rel.shape[0]}"
            )

        if self.root_pose is None:
            raise RuntimeError("root_pose is not initialized")
        actions = model_action_to_abs_action(
            action_chunk_rel,
            self.root_pose,
            chunk_start_root_rel,
        )
        return (
            actions,
            action_chunk_rel,
            chunk_start_root_rel,
            inference_start_step,
        )

    def update_actions_async(self) -> None:
        while not self.stop_event.is_set():
            with self.actions_lock:
                should_update = (
                    self.previous_inference_start_step is None
                    or self.global_exec_step
                    >= self.previous_inference_start_step + self.config.action_exec_s
                )

            if should_update:
                try:
                    (
                        actions,
                        action_chunk_rel,
                        chunk_start_root_rel,
                        inference_start_step,
                    ) = self.get_actions()
                except Exception as exc:
                    if self.previous_actions is None:
                        self.update_error = exc
                        self.stop_event.set()
                        logger.exception("Initial asynchronous action update failed.")
                        return
                    logger.exception("Asynchronous action update failed; retrying.")
                    sleep(self.config.idle_sleep_s)
                    continue

                with self.actions_lock:
                    self.previous_actions = actions.copy()
                    self.previous_action_chunk_rel = action_chunk_rel.copy()
                    self.previous_chunk_start_root_rel = chunk_start_root_rel.copy()
                    self.previous_inference_start_step = inference_start_step

            sleep(self.config.idle_sleep_s)

    def action_execution_async(self, roll_out_len: int) -> None:
        rate = RateLimiter(frequency=self.config.send_fps)
        frames_sent = 0
        active_chunk_start_step = None
        last_sent_action = None
        pending_interpolation = deque()

        while frames_sent < roll_out_len:
            if self.update_error is not None:
                raise RuntimeError(
                    "Asynchronous update thread failed"
                ) from self.update_error
            if self.stop_event.is_set():
                break

            is_interpolation = len(pending_interpolation) > 0
            if is_interpolation:
                mocap_frame = pending_interpolation.popleft()
            else:
                with self.actions_lock:
                    actions = self.previous_actions
                    chunk_start_step = self.previous_inference_start_step
                    if actions is None or actions.shape[0] == 0:
                        mocap_frame = None
                    else:
                        raw_action_index = self.global_exec_step - chunk_start_step
                        action_index = max(
                            0,
                            min(raw_action_index, actions.shape[0] - 1),
                        )
                        mocap_frame = actions[action_index]

                chunk_changed = (
                    mocap_frame is not None
                    and active_chunk_start_step is not None
                    and chunk_start_step != active_chunk_start_step
                )
                if (
                    chunk_changed
                    and self.config.use_chunk_interpolate
                    and last_sent_action is not None
                ):
                    pending_interpolation.extend(
                        interpolate_chunk_transition(
                            last_sent_action,
                            mocap_frame,
                            self.config.chunk_interp_frames,
                        )
                    )
                if mocap_frame is not None:
                    active_chunk_start_step = chunk_start_step
                if pending_interpolation:
                    mocap_frame = pending_interpolation.popleft()
                    is_interpolation = True

            if mocap_frame is None:
                rate.sleep()
                continue

            xyz, wxyz = extract_mocap_xyz_and_wxyz(mocap_frame)
            self.publisher.send_msg(fps=self.config.send_fps, xyz=xyz, wxyz=wxyz)
            last_sent_action = mocap_frame.copy()
            if not is_interpolation:
                with self.actions_lock:
                    self.put_current_body_pose_to_history()
                    self.global_exec_step += 1
            frames_sent += 1
            rate.sleep()

    def put_current_body_pose_to_history(self) -> None:
        msg = self.body_pose.get_msg()
        if msg is not None:
            self.state_history.put(msg)

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

        executed_steps = inference_start_step - previous_inference_start_step
        delta_count = min(max(executed_steps, 0), previous_action_chunk_rel.shape[0])
        root_delta = previous_action_chunk_rel[:delta_count, :3].sum(axis=0)
        return previous_chunk_start_root_rel + root_delta


def main() -> None:
    config = tyro.cli(ClientConfig)
    runner = G1Gr00tAsyncRunner(config)
    try:
        runner.start()
        runner.action_execution_async(config.roll_out)
    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    finally:
        runner.stop()
        logger.info("Asynchronous inference stopped.")


if __name__ == "__main__":
    main()

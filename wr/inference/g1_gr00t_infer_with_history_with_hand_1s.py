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
from PIL import Image, ImageDraw, ImageFont
sys.path.append("/home/unitree/liyifan/wr/")
from wr.data_res.camera_old import VideoCapture, CameraGrabber
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
    smooth_pose7_quat_sign,
)
from data_res.utils import (
    CAMERAS_MAP,
    SELECT_11_INDICES,
    get_camera_name,
    standardize_imu,
)
from serve_res.gr00t import server_client

logger = get_logger(__name__)


@dataclass
class ClientConfig:
    host: str = "192.168.123.165"
    port: int = 9003
    timeout_ms: int = 15000  # 15s
    task_description: str = (
        "pick up the cube and bottle into the bowl"
    )
    replay_data_path: Path = Path(
        "/home/mpz/wr_folder/0529/wr/data/ori/test2/data_root_relative_6D.json"
    )
    model_step_fps: float = 50.0
    camera_fps: float = 20.0
    image_history_indices: tuple[int, ...] = (-20, -10, 0)
    send_fps: float = 50
    history_len: int = 50
    action_chunk_size: int = 30
    num_interp: int = 1
    use_interpolate: bool = False
    roll_out: int = 5000
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
    if frame.ndim != 2:
        raise ValueError(
            f"Expected mocap frame shape {expected_shape}, got {frame.shape}"
        )
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
    frame_history=None,
):
    history_state = np.asarray(history_state, dtype=np.float32)
    image_steps = 1 if frame_history is None else len(frame_history)

    video = {
        "ego_view": np.zeros((1, image_steps, 256, 256, 3), dtype=np.uint8),
        "left_wrist_view": np.zeros((1, image_steps, 256, 256, 3), dtype=np.uint8),
        "right_wrist_view": np.zeros((1, image_steps, 256, 256, 3), dtype=np.uint8),
    }

    if frame_history is not None:
        for name in frame_history[0]:
            video[CAMERAS_MAP[name]] = np.stack(
                [
                    load_image_from_path_or_array(np.asarray(frame[name]), (256, 256))[
                        0, 0
                    ]
                    for frame in frame_history
                ],
                axis=0,
            )[None]
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


def read_camera_frames(camera_caps, flush_count=1):
    frame = {}
    for name, camera_cap in camera_caps.items():
        for _ in range(flush_count):
            ok, camera_frame = camera_cap.read()
            if not ok or camera_frame is None:
                raise RuntimeError(f"Failed to read camera frame from {name}")
            frame[name] = camera_frame
    return frame


class MocapDataQueue:
    def __init__(self, maxlen=300):
        self._lock = threading.Lock()
        self._queue = deque(maxlen=maxlen)
        self._last_xyz = None
        self._last_wxyz = None
        self._in_flight = False

    def put(self, xyz, wxyz, poll_interval=0.01):
        xyz = (
            np.asarray(xyz, dtype=np.float32)
            .reshape(MOCAP_NUM_JOINTS, MOCAP_POS_DIM)
            .copy()
        )
        wxyz = (
            np.asarray(wxyz, dtype=np.float32)
            .reshape(MOCAP_NUM_JOINTS, MOCAP_QUAT_DIM)
            .copy()
        )
        while True:
            with self._lock:
                if len(self._queue) < self._queue.maxlen:
                    self._queue.append((xyz, wxyz))
                    return
            sleep(poll_interval)

    def get_next_or_last(self, should_stop=None, poll_interval=0.01):
        has_warned = False

        while True:
            with self._lock:
                if len(self._queue) > 0:
                    return self._queue.popleft()

            if should_stop is not None and should_stop():
                return None

            if not has_warned:
                logger.warning("Mocap queue is empty; waiting for data to be filled.")
                has_warned = True
            sleep(poll_interval)

    def clear(self):
        with self._lock:
            self._queue.clear()

    def size(self):
        with self._lock:
            return len(self._queue)

    def empty(self):
        return self.size() == 0


class StateHistoryQueue:
    def __init__(self, maxlen=50):
        self._lock = threading.Lock()
        self._queue = deque(maxlen=maxlen)

    def put(self, msg: WR_GAE_BodyPose_Msg_V2, hand_state: np.ndarray):
        with self._lock:
            q_np = np.asarray(msg.robot_qpos[:29], dtype=np.float32)
            imu_np = np.asarray(msg.robot_rootquat, dtype=np.float32)
            imu_np = compute_imu_relative(imu_np[None, :], imu_np[None, :])
            imu_np = quaternion_to_rotation_6d(imu_np)[0]
            body_state = np.concatenate([imu_np, q_np], axis=0)
            self._queue.append(np.concatenate([body_state, hand_state], axis=0))

    def get_all(self, poll_interval=0.01):
        has_warned = False
        maxlen = self._queue.maxlen

        while True:
            states = None
            with self._lock:
                if len(self._queue) > 0:
                    states = [state.copy() for state in self._queue]

            if states is not None:
                if len(states) < maxlen:
                    pad_state = states[0]
                    states = [pad_state.copy()] * (maxlen - len(states)) + states
                return np.stack(states, axis=0)

            if not has_warned:
                logger.warning("History queue is empty; waiting for data to be filled.")
                has_warned = True
            sleep(poll_interval)


class ImageHistoryQueue:
    def __init__(self, indices=(-20, -10, 0)):
        self.indices = tuple(indices)
        self._lock = threading.Lock()
        self._queue = deque(maxlen=abs(min(self.indices)) + 1)

    def put(self, frame):
        with self._lock:
            self._queue.append({name: image.copy() for name, image in frame.items()})

    def get(self, poll_interval=0.01):
        has_warned = False
        while True:
            frames = None
            with self._lock:
                if len(self._queue) > 0:
                    frames = list(self._queue)

            if frames is not None:
                if len(frames) < self._queue.maxlen:
                    frames = [frames[0]] * (self._queue.maxlen - len(frames)) + frames
                return [
                    frames[-1] if index == 0 else frames[index - 1]
                    for index in self.indices
                ]

            if not has_warned:
                logger.warning("Image history queue is empty; waiting for data.")
                has_warned = True
            sleep(poll_interval)


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
        self.rate = RateLimiter(frequency=self.fps)

    def run(self):
        while self.running:
            item = self.mocap_queue.get_next_or_last(
                should_stop=lambda: not self.running
            )
            if item is None:
                break
            xyz, wxyz = item
            try:
                self.publisher.send_msg(fps=self.fps, xyz=xyz, wxyz=wxyz)
                self.rate.sleep()
            except Exception:
                logger.exception("Mocap sender failed; stopping sender thread.")
                self.running = False
                break

    def stop(self):
        self.running = False


def model_action_to_abs_action(
    action_output: np.ndarray, init_pose: np.ndarray, root_rel_cum: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    action_output = np.asarray(action_output, dtype=np.float32)
    logger.info("action output shape: %s", action_output.shape)

    action_with_root_delta = action_output
    root_delta = action_with_root_delta[:, :3]
    action_without_root_delta = action_with_root_delta[:, 3:102]
    action_hand = action_with_root_delta[:, 102:]

    action_mocap_xyz = action_without_root_delta[:, :33].reshape(-1, 11, 3)
    action_mocap_6d = action_without_root_delta[:, 33:].reshape(-1, 11, 6)
    action_mocap = np.concatenate([action_mocap_xyz, action_mocap_6d], axis=-1).reshape(-1, 99)
    action_mocap = np.concatenate([root_delta, action_mocap], axis=-1)

    if root_rel_cum is None:
        action_11x9, root_rel_cum = restore_mocap_from_root_relative(action_mocap)
    else:
        action_11x9, root_rel_cum = restore_mocap_from_root_relative(action_mocap, root_rel_cum)

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
    return ans, root_rel_cum, action_hand


def interpolate_hand_joint(hand_joint_seq: np.ndarray, num_interp: int) -> np.ndarray:
    hand_joint_seq = np.asarray(hand_joint_seq, dtype=np.float32)
    if hand_joint_seq.ndim != 2 or hand_joint_seq.shape[1] != 12:
        raise ValueError(f"Expected hand joint shape (T, 12), got {hand_joint_seq.shape}")
    if hand_joint_seq.shape[0] <= 1 or num_interp <= 0:
        return hand_joint_seq

    steps = np.linspace(0.0, 1.0, num_interp + 2, dtype=np.float32)[:-1]
    interpolated = (
        (1.0 - steps[None, :, None]) * hand_joint_seq[:-1, None, :]
        + steps[None, :, None] * hand_joint_seq[1:, None, :]
    )
    return np.concatenate(
        [interpolated.reshape(-1, 12), hand_joint_seq[-1:]], axis=0
    )


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
        hand_state = hand_subscriber.get_state()
        if hand_state is not None:
            return hand_state
        sleep(poll_interval_s)


if __name__ == "__main__":
    config = tyro.cli(ClientConfig)
    if config.num_interp < 0:
        raise ValueError(f"num_interp must be non-negative, got {config.num_interp}")
    if config.send_fps <= 0:
        raise ValueError(f"send_fps must be positive, got {config.send_fps}")
    if config.model_step_fps <= 0:
        raise ValueError(f"model_step_fps must be positive, got {config.model_step_fps}")
    if config.camera_fps <= 0:
        raise ValueError(f"camera_fps must be positive, got {config.camera_fps}")
    if not config.image_history_indices or 0 not in config.image_history_indices:
        raise ValueError("image_history_indices must contain 0")
    state_sample_every = config.num_interp + 1 if config.use_interpolate else 1

    client = server_client.PolicyClient(
        host=config.host,
        port=config.port,
        timeout_ms=config.timeout_ms,
    )
    print("Waiting for gr00t server to ping")

    if client.ping():
        print("Server is alive!")
    else:
        print("Failed to connect to the server.")
        sys.exit(1)

    body_pose = BodyPoseSubscriberV3(config.body_pose_cfg)
    state_history_queue = StateHistoryQueue(maxlen=config.history_len)
    image_history_queue = ImageHistoryQueue(config.image_history_indices)
    print("[INFO] init camera")
    camera_name_list = get_camera_name(config.camera_config)
    camera_caps = {name: VideoCapture(name) for name in camera_name_list}
    camera_grabber = CameraGrabber(camera_caps)
    camera_grabber.start()
    camera_grabber.wait_until_ready()

    hand_subscriber = MocapUEHandSubscriber()
    hand_publisher = MocapUEHandPublisher()
    body_publisher = MocapUE5G115MsgPublisher(config.mocap_cfg, config.send_fps)
    send_rate = RateLimiter(frequency=config.send_fps)
    
    initial_body_pose_msg = wait_for_body_pose_msg(body_pose)
    initial_hand_state = wait_for_hand_state_msg(hand_subscriber)
    state_history_queue.put(initial_body_pose_msg, initial_hand_state)
    image_history_queue.put(camera_grabber.get_frames())
    while True:
        root_pose = body_pose.get_root_pose()
        if root_pose is not None:
            print("root pose received!")
            break
        sleep(0.01)
    root_pose[2] = 1.0

    try:
        root_rel_cum = None
        last_action = None
        last_hand_action = None
        global_state_sample_every = 0
        image_sample_acc = 0.0
        image_sample_ratio = config.camera_fps / config.model_step_fps
        for idx in range(config.roll_out):

            state_history = state_history_queue.get_all()
            frame_history = image_history_queue.get()
            observation = build_observation_from_msg_with_history(
                state_history,
                config.task_description,
                frame_history=frame_history,
            )
            action_chunk_rel = client.get_action(observation)[0]["mocap"][0].reshape(-1, 114)[:config.action_chunk_size]
            # ret = client.get_action(observation)[0]
            # action_chunk_rel = np.concatenate((ret["root_delta"][0], ret["mocap_xyz"][0], ret["mocap_rot6d"][0]), axis=-1).reshape(-1)
            action_chunk_abs, root_rel_cum, action_hand = model_action_to_abs_action(
                action_chunk_rel, root_pose, root_rel_cum
            )

            if last_action is not None and last_hand_action is not None:
                action_chunk_abs = np.concatenate(
                    (last_action, action_chunk_abs), axis=0
                )
                action_chunk_abs = smooth_pose7_quat_sign(action_chunk_abs)
                action_hand = np.concatenate([last_hand_action, action_hand], axis=0)
                
            if config.use_interpolate:
                action_chunk_abs = interpolate_pose7(
                    action_chunk_abs, config.num_interp
                )
                action_hand = interpolate_hand_joint(action_hand, config.num_interp)

            if last_action is not None:
                action_chunk_abs = action_chunk_abs[1:]
                action_hand = action_hand[1:]

            if action_chunk_abs.shape[0] != action_hand.shape[0]:
                raise ValueError(
                    "Mocap and hand action lengths do not match: "
                    f"{action_chunk_abs.shape[0]} vs {action_hand.shape[0]}"
                )

            for t in range(action_chunk_abs.shape[0]):
                mocap_frame = action_chunk_abs[t]
                xyz, wxyz = extract_mocap_xyz_and_wxyz(mocap_frame)

                body_publisher.send_msg(xyz=xyz, wxyz=wxyz)
                hand_publisher.send(action_hand[t])
                send_rate.sleep()

                global_state_sample_every += 1
                if global_state_sample_every >= state_sample_every:
                    hist_msg = body_pose.get_msg()
                    hist_hand = hand_subscriber.get_state()
                    if hist_msg is not None and hist_hand is not None:
                        state_history_queue.put(hist_msg, hist_hand)
                    image_sample_acc += image_sample_ratio
                    if image_sample_acc >= 1.0:
                        image_history_queue.put(camera_grabber.get_frames())
                        image_sample_acc -= 1.0
                    global_state_sample_every = 0

            last_action = action_chunk_abs[-1:]
            last_hand_action = action_hand[-1:]

    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    finally:
        camera_grabber.stop()
        for camera_cap in camera_caps.values():
            camera_cap.release()
        logger.info("Sender thread stopped.")

import json
import os
import sys
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from time import sleep

import numpy as np
import tyro
from PIL import Image
from loop_rate_limiters import RateLimiter
from scipy.spatial.transform import Rotation
import sys
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
    load_mocap_imu_data
)
from serve_res.gr00t import server_client

logger = get_logger(__name__)


@dataclass
class ClientConfig:
    host: str = "192.168.123.163"
    port: int = 9002
    timeout_ms: int = 15000  # 15s
    task_description: str = (
        "pick up the cube and bottle into the bowl"
    )
    replay_data_path: Path = Path(
        "/home/unitree/mpz/0610/wr_new/data/data.npy"
    )
    camera_fps: float = 20.0
    send_fps: float = 50
    history_len: int = 50
    action_chunk_size: int = 30
    # Mocap command shaping. No frames are inserted or interpolated: every policy
    # action is still sent once at send_fps. Units are m/s, m/s^2, m/s^3 and
    # rad/s, rad/s^2, rad/s^3 respectively.
    enable_command_limiter: bool = True
    # Calibrated from G:\project\data.json after resampling its real mocap
    # demonstration to the 50 Hz deployment rate. These values preserve almost
    # all nominal upper-body motion while constraining abrupt command changes.
    max_linear_speed: float = 0.9
    max_linear_acceleration: float = 18.0
    max_linear_jerk: float = 1500.0
    max_angular_speed: float = 3.5
    max_angular_acceleration: float = 120.0
    max_angular_jerk: float = 10000.0
    # Record every action frame actually sent by the publishers.
    save_actions: bool = False
    action_output_path: Path = Path("/output/action.json")
    roll_out: int = 20000
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
                np.asarray(image), (256, 256), debug=False, name=name
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


class StateHistoryQueue:
    def __init__(self, maxlen=50):
        self._lock = threading.Lock()
        self._queue = deque(maxlen=maxlen)

    def put(self, msg: WR_GAE_BodyPose_Msg_V2, hand_state: np.ndarray):
        with self._lock:
            q_np = np.asarray(msg.robot_qpos[:29], dtype=np.float32)
            imu_np = np.asarray(msg.robot_rootquat[:4], dtype=np.float32)
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


class ExecutedActionRecorder:
    """Stream executed actions to a temporary JSON file and publish it on exit."""

    def __init__(self, enabled: bool, output_path: Path):
        self.enabled = enabled
        self.output_path = Path(output_path)
        self.temp_path = self.output_path.with_name(f"{self.output_path.name}.tmp")
        self._file = None
        self._num_actions = 0
        self._failed = False

    @property
    def num_actions(self) -> int:
        return self._num_actions

    def start(self):
        if not self.enabled:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.temp_path.open("w", encoding="utf-8")
        try:
            self._file.write("[\n")
        except BaseException:
            self._failed = True
            raise

    def record(
        self,
        mocap_action: np.ndarray,
        hand_action: np.ndarray,
        is_policy_step: bool,
    ):
        if not self.enabled:
            return
        if self._file is None:
            raise RuntimeError("Action recorder has not been started")

        serialized_action = json.dumps(
            {
                "mocap": np.asarray(mocap_action, dtype=np.float32).tolist(),
                "hand": np.asarray(hand_action, dtype=np.float32).tolist(),
                "is_policy_step": bool(is_policy_step),
            },
            allow_nan=False,
            separators=(",", ":"),
        )
        record_start = self._file.tell()
        prefix = ",\n" if self._num_actions > 0 else ""
        try:
            self._file.write(prefix + serialized_action)
        except BaseException:
            self._failed = True
            try:
                self._file.seek(record_start)
                self._file.truncate()
            except Exception:
                pass
            else:
                self._failed = False
            raise
        self._num_actions += 1

    def close(self):
        if not self.enabled or self._file is None:
            return

        file_obj = self._file
        self._file = None
        if self._failed:
            file_obj.close()
            logger.error(
                "Action recording failed; incomplete temporary file was not "
                "published: %s",
                self.temp_path,
            )
            return

        try:
            file_obj.write("\n]\n")
            file_obj.flush()
            os.fsync(file_obj.fileno())
        finally:
            file_obj.close()

        os.replace(self.temp_path, self.output_path)
        logger.info(
            "Saved %d executed actions to %s",
            self._num_actions,
            self.output_path,
        )


def model_action_to_abs_action(
    action_output: np.ndarray, init_pose: np.ndarray, root_rel_cum: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    action_output = np.asarray(action_output, dtype=np.float32)
    logger.info("action output shape: %s", action_output.shape)

    action_with_root_delta = action_output.reshape(-1, 114)
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


def clip_vector_norm(values: np.ndarray, maximum: float) -> np.ndarray:
    """Limit vectors along the final axis without changing their direction."""
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    scale = np.minimum(1.0, maximum / np.maximum(norms, 1e-12))
    return values * scale


def count_vectors_over_limit(values: np.ndarray, maximum: float) -> int:
    """Return how many vectors would be changed by ``clip_vector_norm``."""
    return int(np.count_nonzero(np.linalg.norm(values, axis=-1) > maximum))


class MocapCommandLimiter:
    """Rate-limit raw mocap commands by speed, acceleration, and jerk.

    This class never inserts frames or interpolates between policy actions.  Its
    state is the actual command sent on the preceding 50 Hz cycle, not a robot
    joint-state estimate.
    """

    def __init__(self, config: ClientConfig):
        self.dt = 1.0 / config.send_fps
        self.max_linear_speed = config.max_linear_speed
        self.max_linear_acceleration = config.max_linear_acceleration
        self.max_linear_jerk = config.max_linear_jerk
        self.max_angular_speed = config.max_angular_speed
        self.max_angular_acceleration = config.max_angular_acceleration
        self.max_angular_jerk = config.max_angular_jerk
        self.last_pose7: np.ndarray | None = None
        self.linear_velocity = np.zeros((MOCAP_NUM_JOINTS, 3), dtype=np.float64)
        self.linear_acceleration = np.zeros((MOCAP_NUM_JOINTS, 3), dtype=np.float64)
        self.angular_velocity = np.zeros((MOCAP_NUM_JOINTS, 3), dtype=np.float64)
        self.angular_acceleration = np.zeros((MOCAP_NUM_JOINTS, 3), dtype=np.float64)
        self.command_index = 0

    def step(self, target_pose7: np.ndarray) -> np.ndarray:
        expected_shape = (MOCAP_NUM_JOINTS, MOCAP_POS_DIM + MOCAP_QUAT_DIM)
        target_pose7 = np.asarray(target_pose7, dtype=np.float64)
        if target_pose7.shape != expected_shape:
            raise ValueError(f"Expected mocap frame shape {expected_shape}, got {target_pose7.shape}")
        target_pose7 = target_pose7.copy()
        target_pose7[:, 3:] = normalize_quaternion(target_pose7[:, 3:])

        if self.last_pose7 is None:
            output = smooth_pose7_quat_sign(target_pose7[None, ...])[0]
            self.last_pose7 = output.astype(np.float64)
            self.command_index += 1
            return output

        target_linear_velocity = (target_pose7[:, :3] - self.last_pose7[:, :3]) / self.dt
        desired_linear_acceleration = (target_linear_velocity - self.linear_velocity) / self.dt
        desired_linear_jerk = (desired_linear_acceleration - self.linear_acceleration) / self.dt
        linear_jerk_limited = count_vectors_over_limit(
            desired_linear_jerk, self.max_linear_jerk
        )
        limited_linear_jerk = clip_vector_norm(desired_linear_jerk, self.max_linear_jerk)
        next_linear_acceleration = self.linear_acceleration + limited_linear_jerk * self.dt
        linear_acceleration_limited = count_vectors_over_limit(
            next_linear_acceleration, self.max_linear_acceleration
        )
        self.linear_acceleration = clip_vector_norm(
            next_linear_acceleration,
            self.max_linear_acceleration,
        )
        next_linear_velocity = self.linear_velocity + self.linear_acceleration * self.dt
        linear_speed_limited = count_vectors_over_limit(
            next_linear_velocity, self.max_linear_speed
        )
        self.linear_velocity = clip_vector_norm(
            next_linear_velocity,
            self.max_linear_speed,
        )
        output_xyz = self.last_pose7[:, :3] + self.linear_velocity * self.dt

        current_rotation = Rotation.from_quat(np.roll(self.last_pose7[:, 3:], -1, axis=-1))
        target_rotation = Rotation.from_quat(np.roll(target_pose7[:, 3:], -1, axis=-1))
        target_angular_velocity = (current_rotation.inv() * target_rotation).as_rotvec() / self.dt
        desired_angular_acceleration = (target_angular_velocity - self.angular_velocity) / self.dt
        desired_angular_jerk = (desired_angular_acceleration - self.angular_acceleration) / self.dt
        angular_jerk_limited = count_vectors_over_limit(
            desired_angular_jerk, self.max_angular_jerk
        )
        limited_angular_jerk = clip_vector_norm(desired_angular_jerk, self.max_angular_jerk)
        next_angular_acceleration = self.angular_acceleration + limited_angular_jerk * self.dt
        angular_acceleration_limited = count_vectors_over_limit(
            next_angular_acceleration, self.max_angular_acceleration
        )
        self.angular_acceleration = clip_vector_norm(
            next_angular_acceleration,
            self.max_angular_acceleration,
        )
        next_angular_velocity = self.angular_velocity + self.angular_acceleration * self.dt
        angular_speed_limited = count_vectors_over_limit(
            next_angular_velocity, self.max_angular_speed
        )
        self.angular_velocity = clip_vector_norm(
            next_angular_velocity,
            self.max_angular_speed,
        )
        output_rotation = current_rotation * Rotation.from_rotvec(self.angular_velocity * self.dt)

        output = np.empty_like(target_pose7)
        output[:, :3] = output_xyz
        output[:, 3:] = np.roll(output_rotation.as_quat(), 1, axis=-1)
        output = smooth_pose7_quat_sign(output[None, ...])[0]
        self.last_pose7 = output.astype(np.float64)
        if any(
            (
                linear_jerk_limited,
                linear_acceleration_limited,
                linear_speed_limited,
                angular_jerk_limited,
                angular_acceleration_limited,
                angular_speed_limited,
            )
        ):
            logger.warning(
                "Mocap command %d rate-limited: linear(jerk=%d, accel=%d, speed=%d), "
                "angular(jerk=%d, accel=%d, speed=%d) out of %d points",
                self.command_index,
                linear_jerk_limited,
                linear_acceleration_limited,
                linear_speed_limited,
                angular_jerk_limited,
                angular_acceleration_limited,
                angular_speed_limited,
                MOCAP_NUM_JOINTS,
            )
        self.command_index += 1
        return output


def validate_raw_action_chunk(
    mocap_chunk: np.ndarray, hand_chunk: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Validate raw, one-policy-step-per-send action chunks without interpolation."""
    mocap_chunk = np.asarray(mocap_chunk, dtype=np.float32)
    hand_chunk = np.asarray(hand_chunk, dtype=np.float32)
    expected_shape = (MOCAP_NUM_JOINTS, MOCAP_POS_DIM + MOCAP_QUAT_DIM)
    if mocap_chunk.ndim != 3 or mocap_chunk.shape[1:] != expected_shape:
        raise ValueError(
            "Expected raw mocap chunk shape "
            f"(T, {expected_shape[0]}, {expected_shape[1]}), got {mocap_chunk.shape}"
        )
    if hand_chunk.ndim != 2 or hand_chunk.shape[1] != 12:
        raise ValueError(f"Expected raw hand chunk shape (T, 12), got {hand_chunk.shape}")
    if mocap_chunk.shape[0] == 0 or mocap_chunk.shape[0] != hand_chunk.shape[0]:
        raise ValueError(
            "Raw mocap and hand chunks must be non-empty and have equal lengths, got "
            f"{mocap_chunk.shape[0]} and {hand_chunk.shape[0]}"
        )
    return mocap_chunk, hand_chunk


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
    if config.send_fps <= 0:
        raise ValueError(f"send_fps must be positive, got {config.send_fps}")
    positive_limits = {
        "max_linear_speed": config.max_linear_speed,
        "max_linear_acceleration": config.max_linear_acceleration,
        "max_linear_jerk": config.max_linear_jerk,
        "max_angular_speed": config.max_angular_speed,
        "max_angular_acceleration": config.max_angular_acceleration,
        "max_angular_jerk": config.max_angular_jerk,
    }
    for name, value in positive_limits.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")

    client = server_client.PolicyClient(
        host=config.host,
        port=config.port,
        timeout_ms=config.timeout_ms,
    )
    print("Waiting for gr00t server to ping")

    # if client.ping():
    #     print("Server is alive!")
    # else:
    #     print("Failed to connect to the server.")
    #     sys.exit(1)

    body_pose = BodyPoseSubscriberV3(config.body_pose_cfg)
    state_history_queue = StateHistoryQueue(maxlen=config.history_len)
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
    command_limiter = MocapCommandLimiter(config)

    while True:
        root_pose = body_pose.get_root_pose()
        if root_pose is not None:
            print("root pose received!")
            break
        sleep(0.01)


    # replay_data = load_mocap_imu_data(config.replay_data_path)
    # _MOCAP_POSE_DIM = _MOCAP_POS_DIM + _MOCAP_QUAT_DIM
    # assert replay_data.ndim == 3, (
    #     f"Expected (T, {_MOCAP_NUM_JOINTS}, {_MOCAP_POSE_DIM}), got {replay_data.shape}"
    # )
    # assert replay_data.shape[1] == _MOCAP_NUM_JOINTS, (
    #     f"Expected {_MOCAP_NUM_JOINTS} joints, got {replay_data.shape}"
    # )
    # assert replay_data.shape[2] == _MOCAP_POSE_DIM, (
    #     f"Expected pose dim {_MOCAP_POSE_DIM}, got {replay_data.shape}"
    # )
    # logger.info(f"The replay episode length is: {replay_data.shape[0]}")

    # replay_data_init = replay_data[0:1, :, :].copy()
    # num_frames, num_joints, poses = replay_data_init.shape
    # root_pose_tiled = np.tile(root_pose, (num_frames * num_joints, 1))
    # replay_data_init_flat = replay_data_init.reshape(num_frames * num_joints, -1)
    # replay_data_init = compute_absolute(root_pose_tiled, replay_data_init_flat)
    # replay_data_init = replay_data_init.reshape(num_frames, num_joints, poses)[0]


    # replay_data_init = replay_data[0, :, :].copy()

    # print(f"replay_data_init: {replay_data_init.shape}")


    # body_publisher.send_msg(xyz=replay_data_init[:, 0:3], wxyz=replay_data_init[:, 3:7])
    # hand_publisher.send([0]*12)


    logger.info("MocapSender thread started.")
    sleep(1)

    initial_body_pose_msg = wait_for_body_pose_msg(body_pose)
    # initial_body_root_pose = body_pose.get_root_pose()
    # print(f"initial_body_root_pose: {initial_body_root_pose}")
    initial_hand_state = wait_for_hand_state_msg(hand_subscriber)

    state_history_queue.put(initial_body_pose_msg, initial_hand_state)
    # root_pose = np.concatenate(
    #     [
    #         np.asarray(initial_body_pose_msg.robot_xpos[36:39], dtype=np.float32),
    #         np.asarray(initial_body_pose_msg.robot_rootquat, dtype=np.float32),
    #     ]
    # )
    # root_pose = root_pose
    print(f"root_pose : {root_pose}")
    root_pose[2] = 1.0


    action_recorder = ExecutedActionRecorder(
        enabled=config.save_actions,
        output_path=config.action_output_path,
    )

    try:
        action_recorder.start()
        root_rel_cum = None
        for idx in range(config.roll_out):

            frame = camera_grabber.get_frames()

            state_history = state_history_queue.get_all()
            observation = build_observation_from_msg_with_history(
                state_history,
                config.task_description,
                frame=frame,
            )

            action_chunk_rel = client.get_action(observation)[0]["mocap"][0]
            # ret = client.get_action(observation)[0]
            # action_chunk_rel = np.concatenate((ret["root_delta"][0], ret["mocap_xyz"][0], ret["mocap_rot6d"][0]), axis=-1).reshape(-1)
            action_chunk_abs, root_rel_cum, action_hand = model_action_to_abs_action(
                action_chunk_rel, root_pose, root_rel_cum
            )

            action_chunk_abs, action_hand = validate_raw_action_chunk(
                action_chunk_abs, action_hand
            )

            for t in range(action_chunk_abs.shape[0]):
                raw_mocap_frame = action_chunk_abs[t]
                mocap_frame = (
                    command_limiter.step(raw_mocap_frame)
                    if config.enable_command_limiter
                    else raw_mocap_frame
                )
                xyz, wxyz = extract_mocap_xyz_and_wxyz(mocap_frame)
                executed_mocap_frame = np.concatenate([xyz, wxyz], axis=-1)
                 
                body_publisher.send_msg(xyz=xyz, wxyz=wxyz)
                hand_publisher.send(action_hand[t])
                action_recorder.record(
                    mocap_action=executed_mocap_frame,
                    hand_action=action_hand[t],
                    is_policy_step=True,
                )
                print(action_hand[t])
                send_rate.sleep()

                hist_msg = body_pose.get_msg()
                hist_hand = hand_subscriber.get_state()
                if hist_msg is not None and hist_hand is not None:
                    state_history_queue.put(hist_msg, hist_hand)

    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    finally:
        try:
            action_recorder.close()
        finally:
            camera_grabber.stop()
            for camera_cap in camera_caps.values():
                camera_cap.release()
            logger.info("Sender thread stopped.")

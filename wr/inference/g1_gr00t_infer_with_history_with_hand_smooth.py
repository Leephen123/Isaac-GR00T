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
from scipy.spatial.transform import Rotation, Slerp
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
    # Frames inserted between adjacent policy actions inside a chunk.
    enable_intra_chunk_interp: bool = False
    intra_chunk_interp_num: int = 2
    # Frames inserted between the previous chunk endpoint and current chunk start.
    # This legacy option lengthens a chunk. Prefer chunk_boundary_blend_frames below
    # when preserving the policy's original 50 Hz timing is important.
    enable_inter_chunk_interp: bool = False
    inter_chunk_interp_num: int = 2
    # Blend the start of each new mocap chunk with the final command of the prior
    # chunk. This changes no frame count and does not modify dexterous-hand actions.
    enable_chunk_boundary_blend: bool = True
    chunk_boundary_blend_frames: int = 4
    # Record every action frame actually sent by the publishers.
    save_actions: bool = False
    action_output_path: Path = Path("/output/action_pred_smooth.json")
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


def blend_pose7_frame(
    start_pose7: np.ndarray,
    end_pose7: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Blend two mocap frames with linear xyz interpolation and quaternion SLERP."""
    expected_shape = (MOCAP_NUM_JOINTS, MOCAP_POS_DIM + MOCAP_QUAT_DIM)
    start_pose7 = np.asarray(start_pose7, dtype=np.float64)
    end_pose7 = np.asarray(end_pose7, dtype=np.float64)
    if start_pose7.shape != expected_shape or end_pose7.shape != expected_shape:
        raise ValueError(
            "Expected mocap frames with shape "
            f"{expected_shape}, got {start_pose7.shape} and {end_pose7.shape}"
        )
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    blended = np.empty_like(start_pose7)
    blended[:, :3] = (1.0 - alpha) * start_pose7[:, :3] + alpha * end_pose7[:, :3]

    start_wxyz = normalize_quaternion(start_pose7[:, 3:])
    end_wxyz = normalize_quaternion(end_pose7[:, 3:])
    # q and -q encode the same orientation. Select the nearby representation so
    # every SLERP follows the short, continuous arc at a chunk boundary.
    end_wxyz = end_wxyz.copy()
    end_wxyz[np.sum(start_wxyz * end_wxyz, axis=-1) < 0.0] *= -1.0
    for joint_idx in range(MOCAP_NUM_JOINTS):
        key_rotations = Rotation.from_quat(
            np.stack(
                [
                    np.roll(start_wxyz[joint_idx], -1),
                    np.roll(end_wxyz[joint_idx], -1),
                ]
            )
        )
        quat_xyzw = Slerp([0.0, 1.0], key_rotations)([alpha]).as_quat()[0]
        blended[joint_idx, 3:] = np.roll(quat_xyzw, 1)
    return blended.astype(np.float32)


def blend_chunk_boundary(
    prev_last_pose7: np.ndarray | None,
    cur_chunk_pose7: np.ndarray,
    blend_frames: int,
) -> np.ndarray:
    """Smooth a chunk boundary in place without inserting or removing frames.

    The first ``blend_frames`` output commands are moved continuously from the
    last *sent* mocap command towards their respective policy predictions. A
    smoothstep schedule reaches the unmodified policy trajectory at the final
    blended frame, preserving the chunk length and send rate.
    """
    cur_chunk_pose7 = np.asarray(cur_chunk_pose7, dtype=np.float32).copy()
    if blend_frames < 0:
        raise ValueError(f"blend_frames must be non-negative, got {blend_frames}")
    if prev_last_pose7 is None or blend_frames == 0:
        return cur_chunk_pose7
    if cur_chunk_pose7.ndim != 3 or cur_chunk_pose7.shape[1:] != (
        MOCAP_NUM_JOINTS,
        MOCAP_POS_DIM + MOCAP_QUAT_DIM,
    ):
        raise ValueError(
            "Expected current mocap chunk shape "
            f"(T, {MOCAP_NUM_JOINTS}, {MOCAP_POS_DIM + MOCAP_QUAT_DIM}), "
            f"got {cur_chunk_pose7.shape}"
        )

    frame_count = min(blend_frames, cur_chunk_pose7.shape[0])
    for frame_idx in range(frame_count):
        linear_alpha = (frame_idx + 1) / frame_count
        alpha = linear_alpha * linear_alpha * (3.0 - 2.0 * linear_alpha)
        cur_chunk_pose7[frame_idx] = blend_pose7_frame(
            prev_last_pose7,
            cur_chunk_pose7[frame_idx],
            alpha,
        )
    return smooth_pose7_quat_sign(cur_chunk_pose7)


def build_action_frames(
    prev_last_pose7: np.ndarray | None,
    prev_last_hand: np.ndarray | None,
    cur_chunk_pose7: np.ndarray,
    cur_chunk_hand: np.ndarray,
    enable_intra_chunk_interp: bool,
    intra_chunk_interp_num: int,
    enable_inter_chunk_interp: bool,
    inter_chunk_interp_num: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build synchronized action frames and mark the original policy steps."""

    if intra_chunk_interp_num < 0:
        raise ValueError(
            "intra_chunk_interp_num must be non-negative, "
            f"got {intra_chunk_interp_num}"
        )
    if inter_chunk_interp_num < 0:
        raise ValueError(
            "inter_chunk_interp_num must be non-negative, "
            f"got {inter_chunk_interp_num}"
        )
    if (prev_last_pose7 is None) != (prev_last_hand is None):
        raise ValueError(
            "prev_last_pose7 and prev_last_hand must both be provided or both be None"
        )

    cur_chunk_pose7 = np.asarray(cur_chunk_pose7, dtype=np.float32)
    cur_chunk_hand = np.asarray(cur_chunk_hand, dtype=np.float32)
    expected_pose_shape = (
        MOCAP_NUM_JOINTS,
        MOCAP_POS_DIM + MOCAP_QUAT_DIM,
    )
    if cur_chunk_pose7.ndim != 3 or cur_chunk_pose7.shape[1:] != expected_pose_shape:
        raise ValueError(
            "Expected current mocap chunk shape "
            f"(T, {expected_pose_shape[0]}, {expected_pose_shape[1]}), "
            f"got {cur_chunk_pose7.shape}"
        )
    if cur_chunk_hand.ndim != 2 or cur_chunk_hand.shape[1] != 12:
        raise ValueError(
            f"Expected current hand chunk shape (T, 12), got {cur_chunk_hand.shape}"
        )
    if cur_chunk_pose7.shape[0] == 0:
        raise ValueError("Cannot execute an empty action chunk")
    if cur_chunk_pose7.shape[0] != cur_chunk_hand.shape[0]:
        raise ValueError(
            "Mocap and hand action lengths do not match before interpolation: "
            f"{cur_chunk_pose7.shape[0]} vs {cur_chunk_hand.shape[0]}"
        )

    prev_pose = None
    prev_hand = None
    if prev_last_pose7 is not None and prev_last_hand is not None:
        prev_pose = np.asarray(prev_last_pose7, dtype=np.float32)
        prev_hand = np.asarray(prev_last_hand, dtype=np.float32)
        if prev_pose.shape != expected_pose_shape:
            raise ValueError(
                f"Expected previous mocap action shape {expected_pose_shape}, "
                f"got {prev_pose.shape}"
            )
        if prev_hand.shape != (12,):
            raise ValueError(
                f"Expected previous hand action shape (12,), got {prev_hand.shape}"
            )

        smoothed_pose7 = smooth_pose7_quat_sign(
            np.concatenate([prev_pose[None, ...], cur_chunk_pose7], axis=0)
        )
        prev_pose = smoothed_pose7[0]
        cur_chunk_pose7 = smoothed_pose7[1:]
    else:
        cur_chunk_pose7 = smooth_pose7_quat_sign(cur_chunk_pose7)

    if enable_intra_chunk_interp and intra_chunk_interp_num > 0:
        action_frames = interpolate_pose7(
            cur_chunk_pose7,
            intra_chunk_interp_num,
        )
        hand_frames = interpolate_hand_joint(
            cur_chunk_hand,
            intra_chunk_interp_num,
        )
        policy_step_mask = np.zeros(action_frames.shape[0], dtype=bool)
        policy_step_mask[::intra_chunk_interp_num + 1] = True
    else:
        action_frames = cur_chunk_pose7
        hand_frames = cur_chunk_hand
        policy_step_mask = np.ones(action_frames.shape[0], dtype=bool)

    bridge_frame_count = 0
    if (
        prev_pose is not None
        and prev_hand is not None
        and enable_inter_chunk_interp
        and inter_chunk_interp_num > 0
    ):
        bridge_frames = interpolate_pose7(
            np.stack([prev_pose, cur_chunk_pose7[0]], axis=0),
            inter_chunk_interp_num,
        )[1:-1]
        bridge_hand_frames = interpolate_hand_joint(
            np.stack([prev_hand, cur_chunk_hand[0]], axis=0),
            inter_chunk_interp_num,
        )[1:-1]
        bridge_frame_count = bridge_frames.shape[0]
        action_frames = np.concatenate([bridge_frames, action_frames], axis=0)
        hand_frames = np.concatenate([bridge_hand_frames, hand_frames], axis=0)
        policy_step_mask = np.concatenate(
            [
                np.zeros(bridge_frame_count, dtype=bool),
                policy_step_mask,
            ],
            axis=0,
        )

    if not (
        action_frames.shape[0]
        == hand_frames.shape[0]
        == policy_step_mask.shape[0]
    ):
        raise RuntimeError(
            "Mocap, hand, and policy-step mask lengths do not match after "
            "interpolation: "
            f"{action_frames.shape[0]}, {hand_frames.shape[0]}, "
            f"{policy_step_mask.shape[0]}"
        )

    logger.info(
        "Built action chunk: policy_steps=%d, intra_interp=%d, "
        "inter_interp=%d, total_frames=%d",
        cur_chunk_pose7.shape[0],
        intra_chunk_interp_num if enable_intra_chunk_interp else 0,
        bridge_frame_count,
        action_frames.shape[0],
    )
    return (
        action_frames.astype(np.float32),
        hand_frames.astype(np.float32),
        policy_step_mask,
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
    if config.intra_chunk_interp_num < 0:
        raise ValueError(
            "intra_chunk_interp_num must be non-negative, "
            f"got {config.intra_chunk_interp_num}"
        )
    if config.inter_chunk_interp_num < 0:
        raise ValueError(
            "inter_chunk_interp_num must be non-negative, "
            f"got {config.inter_chunk_interp_num}"
        )
    if config.chunk_boundary_blend_frames < 0:
        raise ValueError(
            "chunk_boundary_blend_frames must be non-negative, "
            f"got {config.chunk_boundary_blend_frames}"
        )
    if config.send_fps <= 0:
        raise ValueError(f"send_fps must be positive, got {config.send_fps}")

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
        last_action = None
        last_hand_action = None
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

            action_chunk_abs, action_hand, policy_step_mask = build_action_frames(
                prev_last_pose7=last_action,
                prev_last_hand=last_hand_action,
                cur_chunk_pose7=action_chunk_abs,
                cur_chunk_hand=action_hand,
                enable_intra_chunk_interp=config.enable_intra_chunk_interp,
                intra_chunk_interp_num=config.intra_chunk_interp_num,
                enable_inter_chunk_interp=config.enable_inter_chunk_interp,
                inter_chunk_interp_num=config.inter_chunk_interp_num,
            )
            # Unlike inter-chunk interpolation, this does not add bridge frames:
            # the chunk is still sent at exactly 50 Hz for its original duration.
            if config.enable_chunk_boundary_blend:
                action_chunk_abs = blend_chunk_boundary(
                    prev_last_pose7=last_action,
                    cur_chunk_pose7=action_chunk_abs,
                    blend_frames=config.chunk_boundary_blend_frames,
                )

            for t in range(action_chunk_abs.shape[0]):
                mocap_frame = action_chunk_abs[t]
                xyz, wxyz = extract_mocap_xyz_and_wxyz(mocap_frame)
                executed_mocap_frame = np.concatenate([xyz, wxyz], axis=-1)
                 
                body_publisher.send_msg(xyz=xyz, wxyz=wxyz)
                hand_publisher.send(action_hand[t])
                action_recorder.record(
                    mocap_action=executed_mocap_frame,
                    hand_action=action_hand[t],
                    is_policy_step=policy_step_mask[t],
                )
                print(action_hand[t])
                send_rate.sleep()

                if policy_step_mask[t]:
                    hist_msg = body_pose.get_msg()
                    hist_hand = hand_subscriber.get_state()
                    if hist_msg is not None and hist_hand is not None:
                        state_history_queue.put(hist_msg, hist_hand)

            last_action = action_chunk_abs[-1].copy()
            last_hand_action = action_hand[-1].copy()

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

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from time import sleep

import numpy as np
import tyro
from loop_rate_limiters import RateLimiter

sys.path.append("/home/unitree/liyifan/wr/")

from data_res.dds import (
    MOCAP_NUM_JOINTS,
    MOCAP_POS_DIM,
    MOCAP_QUAT_DIM,
    BodyPoseConfig,
    BodyPoseSubscriberV3,
    MocapConfig,
    MocapUE5G115MsgPublisher,
    MocapUEHandPublisher,
)
from data_res.log import get_logger
from data_res.transforms import (
    compute_absolute,
    compute_relative,
    normalize_quaternion,
    restore_mocap_from_root_relative_delta,
    rotation_6d_to_quaternion,
)
from data_res.utils import SELECT_11_INDICES


logger = get_logger(__name__)


@dataclass
class ClientConfig:
    replay_data_path: Path = Path("data_all_points_relative_6D.json")
    send_fps: float = 50.0
    action_chunk_size: int = 50
    roll_out: int = 20000
    mocap_cfg: MocapConfig = field(
        default_factory=lambda: MocapConfig(
            domain_id=1,
            topic_name="MocapUE5G115Topic",
            depth=4,
        )
    )
    body_pose_cfg: BodyPoseConfig = field(
        default_factory=lambda: BodyPoseConfig(
            domain_id=1,
            topic_name="WR/BodyPose",
            depth=4,
        )
    )


def extract_mocap_xyz_and_wxyz(
    frame: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    expected_shape = (
        MOCAP_NUM_JOINTS,
        MOCAP_POS_DIM + MOCAP_QUAT_DIM,
    )

    if frame.shape != expected_shape:
        raise ValueError(
            f"Expected mocap frame shape {expected_shape}, got {frame.shape}"
        )

    xyz = frame[:, :MOCAP_POS_DIM]
    wxyz = normalize_quaternion(frame[:, MOCAP_POS_DIM:])
    return xyz, wxyz


def model_action_to_abs_action(
    action_output: np.ndarray,
    init_pose: np.ndarray,
    root_rel_cum: np.ndarray,
    joint_rel_cum: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    action_output = np.asarray(action_output, dtype=np.float32)
    logger.info("action output shape: %s", action_output.shape)

    action_with_root_delta = action_output.reshape(-1, 114)
    root_delta = action_with_root_delta[:, :3]
    action_without_root_delta = action_with_root_delta[:, 3:102]
    action_hand = action_with_root_delta[:, 102:]

    action_mocap_xyz = action_without_root_delta[:, :33].reshape(-1, 11, 3)
    action_mocap_6d = action_without_root_delta[:, 33:].reshape(-1, 11, 6)
    action_mocap = np.concatenate(
        [action_mocap_xyz, action_mocap_6d], axis=-1
    ).reshape(-1, 99)
    action_mocap = np.concatenate([root_delta, action_mocap], axis=-1)

    action_11x9, root_rel_cum, joint_rel_cum = (
        restore_mocap_from_root_relative_delta(
            action_mocap,
            root_rel_cum,
            joint_rel_cum,
        )
    )

    num_frames = action_11x9.shape[0]
    action_15x9 = np.zeros((num_frames, 15, 9), dtype=np.float32)
    action_15x9[..., 0:3] = 0.0
    action_15x9[..., 3:9] = np.array(
        [1, 0, 0, 0, 1, 0], dtype=np.float32
    )
    action_15x9[:, SELECT_11_INDICES, :] = action_11x9

    action_15x7 = rotation_6d_to_quaternion(action_15x9)
    num_frames, num_joints, num_poses = action_15x7.shape
    action_15x7_flat = action_15x7.reshape(
        num_frames * num_joints, num_poses
    )
    init_pose_x7_flat = np.tile(
        init_pose, (num_frames * num_joints, 1)
    )

    ans = compute_absolute(
        init_pose_x7_flat,
        action_15x7_flat,
    ).reshape(num_frames, num_joints, num_poses)

    return ans, root_rel_cum, joint_rel_cum, action_hand


if __name__ == "__main__":
    config = tyro.cli(ClientConfig)

    if config.send_fps <= 0:
        raise ValueError(
            f"send_fps must be positive, got {config.send_fps}"
        )

    if config.action_chunk_size <= 0:
        raise ValueError(
            "action_chunk_size must be positive, "
            f"got {config.action_chunk_size}"
        )

    action_horizon = config.action_chunk_size

    with config.replay_data_path.open("r", encoding="utf-8") as file:
        replay_data = json.load(file)

    replay_actions = []

    for frame_index, record in enumerate(replay_data):
        mocap = np.asarray(record["mocap"], dtype=np.float32)
        hand_cmd = np.asarray(record["hand_cmd"], dtype=np.float32)

        if mocap.shape != (138,):
            raise ValueError(
                f"Frame {frame_index}: expected mocap shape (138,), "
                f"got {mocap.shape}"
            )

        if hand_cmd.shape != (12,):
            raise ValueError(
                f"Frame {frame_index}: expected hand_cmd shape (12,), "
                f"got {hand_cmd.shape}"
            )

        root_delta = mocap[:3]
        mocap_15x9 = mocap[3:].reshape(15, 9)
        mocap_11x9 = mocap_15x9[SELECT_11_INDICES]

        action = np.concatenate(
            [
                root_delta,
                mocap_11x9[:, :3].reshape(-1),
                mocap_11x9[:, 3:9].reshape(-1),
                hand_cmd,
            ]
        ).astype(np.float32)

        replay_actions.append(action)

    replay_actions = np.stack(replay_actions)
    logger.info("Loaded replay actions: %s", replay_actions.shape)

    body_pose = BodyPoseSubscriberV3(config.body_pose_cfg)
    hand_publisher = MocapUEHandPublisher()
    body_publisher = MocapUE5G115MsgPublisher(
        config.mocap_cfg,
        config.send_fps,
    )
    send_rate = RateLimiter(frequency=config.send_fps)

    while True:
        root_pose = body_pose.get_root_pose()
        if root_pose is not None:
            print("root pose received!")
            break
        sleep(0.01)

    logger.info("MocapSender thread started.")
    sleep(1)

    print(f"root_pose: {root_pose}")
    relative_reference_pose = root_pose.copy()
    root_pose[2] = 1.0

    pose15 = body_pose.get_15_pose7()
    if pose15 is None:
        raise RuntimeError("Failed to receive the initial 15-point body pose")

    root_tiled = np.repeat(
        relative_reference_pose[None, :],
        MOCAP_NUM_JOINTS,
        axis=0,
    )
    pose15_rel = compute_relative(root_tiled, pose15)
    joint_rel_cum = pose15_rel[SELECT_11_INDICES, :3].astype(np.float32)
    joint_rel_cum[0] = 0.0

    try:
        root_rel_cum = np.zeros(3, dtype=np.float32)
        action_index = 0

        for _ in range(config.roll_out):
            if action_index >= len(replay_actions):
                logger.info("All replay actions have been sent.")
                break

            action_chunk_rel = replay_actions[
                action_index : action_index + action_horizon
            ]
            action_index += len(action_chunk_rel)

            (
                action_chunk_abs,
                root_rel_cum,
                joint_rel_cum,
                action_hand,
            ) = model_action_to_abs_action(
                action_chunk_rel,
                root_pose,
                root_rel_cum,
                joint_rel_cum,
            )

            for t in range(action_chunk_abs.shape[0]):
                mocap_frame = action_chunk_abs[t]
                xyz, wxyz = extract_mocap_xyz_and_wxyz(mocap_frame)

                body_publisher.send_msg(xyz=xyz, wxyz=wxyz)
                hand_publisher.send(action_hand[t])
                send_rate.sleep()

    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    finally:
        logger.info("Sender stopped.")

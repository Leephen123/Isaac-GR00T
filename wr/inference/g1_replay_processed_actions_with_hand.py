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
    BodyPoseConfig,
    BodyPoseSubscriberV3,
    MocapConfig,
    MocapUE5G115MsgPublisher,
    MocapUEHandPublisher,
)
from data_res.transforms import (
    compute_absolute,
    compute_relative,
    normalize_quaternion,
    restore_mocap_from_root_relative_delta,
    rotation_6d_to_quaternion,
)


MOCAP_NUM_JOINTS = 15
SELECT_11_INDICES = [0, 2, 3, 6, 7, 9, 10, 11, 12, 13, 14]
SELECT_15_BODY_INDICES = [12, 1, 3, 5, 5, 7, 9, 11, 11, 16, 18, 21, 23, 25, 28]


@dataclass
class ReplayConfig:
    replay_data_path: Path = Path("data_all_points_relative_6D.json")
    send_fps: float = 50.0
    chunk_size: int = 50
    start_frame: int = 0
    max_frames: int = 50  # -1 表示回放到 episode 结束
    countdown_seconds: int = 5
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


def load_actions(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as file:
        records = json.load(file)

    actions = []
    for frame_index, record in enumerate(records):
        mocap = np.asarray(record["mocap"], dtype=np.float32)
        hand_cmd = np.asarray(record["hand_cmd"], dtype=np.float32)

        if mocap.shape != (138,):
            raise ValueError(
                f"Frame {frame_index}: expected mocap shape (138,), got {mocap.shape}"
            )
        if hand_cmd.shape != (12,):
            raise ValueError(
                f"Frame {frame_index}: expected hand_cmd shape (12,), got {hand_cmd.shape}"
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

        if action.shape != (114,):
            raise ValueError(
                f"Frame {frame_index}: expected action shape (114,), got {action.shape}"
            )
        if not np.isfinite(action).all():
            raise ValueError(f"Frame {frame_index}: action contains NaN or infinity")

        actions.append(action)

    return np.stack(actions)


def wait_for_body_pose(body_pose: BodyPoseSubscriberV3):
    while True:
        msg = body_pose.get_msg()
        if msg is not None:
            return msg
        sleep(0.01)


def body_pose_msg_to_pose15(msg) -> np.ndarray:
    if int(msg.robot_joint_num) < 29:
        raise ValueError(f"robot_joint_num={msg.robot_joint_num}, expected at least 29")

    xyz_all = np.asarray(msg.robot_xpos[: 29 * 3], dtype=np.float32).reshape(29, 3)
    wxyz_all = np.asarray(msg.robot_xquat[: 29 * 4], dtype=np.float32).reshape(29, 4)
    selected = np.asarray(SELECT_15_BODY_INDICES, dtype=np.int64)

    xyz = xyz_all[selected].copy()
    wxyz = wxyz_all[selected].copy()
    xyz[0] = xyz_all[12]
    wxyz[0] = np.asarray(msg.robot_rootquat[:4], dtype=np.float32)
    wxyz = normalize_quaternion(wxyz).astype(np.float32)

    return np.concatenate([xyz, wxyz], axis=-1)


def initialize_from_body_pose(msg) -> tuple[np.ndarray, np.ndarray]:
    pose15 = body_pose_msg_to_pose15(msg)
    root_pose = pose15[0].copy()
    root_tiled = np.repeat(root_pose[None, :], MOCAP_NUM_JOINTS, axis=0)
    pose15_relative = compute_relative(root_tiled, pose15)
    joint_rel_cum = pose15_relative[SELECT_11_INDICES, :3].astype(np.float32)
    joint_rel_cum[0] = 0.0
    return root_pose.astype(np.float32), joint_rel_cum


def action_to_world_pose(
    action: np.ndarray,
    root_pose: np.ndarray,
    root_rel_cum: np.ndarray,
    joint_rel_cum: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    action = np.asarray(action, dtype=np.float32).reshape(-1, 114)

    root_delta = action[:, :3]
    joint_xyz_delta = action[:, 3:36].reshape(-1, 11, 3)
    joint_rotation_6d = action[:, 36:102].reshape(-1, 11, 6)
    hand_action = action[:, 102:114]

    mocap_11x9 = np.concatenate(
        [joint_xyz_delta, joint_rotation_6d], axis=-1
    )
    action_102 = np.concatenate(
        [root_delta, mocap_11x9.reshape(len(action), -1)], axis=-1
    )

    restored_11x9, root_rel_cum, joint_rel_cum = (
        restore_mocap_from_root_relative_delta(
            action_102,
            root_rel_cum,
            joint_rel_cum,
        )
    )

    restored_15x9 = np.zeros((len(action), 15, 9), dtype=np.float32)
    restored_15x9[..., 3:9] = np.asarray(
        [1, 0, 0, 0, 1, 0], dtype=np.float32
    )
    restored_15x9[:, SELECT_11_INDICES] = restored_11x9
    restored_15x7 = rotation_6d_to_quaternion(restored_15x9)

    root_pose_tiled = np.broadcast_to(
        root_pose,
        (len(action), 15, 7),
    ).reshape(-1, 7)
    world_pose = compute_absolute(
        root_pose_tiled,
        restored_15x7.reshape(-1, 7),
    ).reshape(len(action), 15, 7)

    return (
        world_pose.astype(np.float32),
        root_rel_cum.astype(np.float32),
        joint_rel_cum.astype(np.float32),
        hand_action,
    )


def main(config: ReplayConfig) -> None:
    if config.send_fps <= 0:
        raise ValueError(f"send_fps must be positive, got {config.send_fps}")
    if config.chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {config.chunk_size}")

    all_actions = load_actions(config.replay_data_path)
    if config.start_frame < 0 or config.start_frame >= len(all_actions):
        raise ValueError(
            f"start_frame must be in [0, {len(all_actions)}), got {config.start_frame}"
        )

    if config.max_frames == -1:
        actions = all_actions[config.start_frame :]
    elif config.max_frames > 0:
        actions = all_actions[
            config.start_frame : config.start_frame + config.max_frames
        ]
    else:
        raise ValueError("max_frames must be -1 or a positive integer")

    print(f"Loaded action shape: {actions.shape}")

    body_pose = BodyPoseSubscriberV3(config.body_pose_cfg)
    body_publisher = MocapUE5G115MsgPublisher(config.mocap_cfg, config.send_fps)
    hand_publisher = MocapUEHandPublisher()
    send_rate = RateLimiter(frequency=config.send_fps)

    print("Waiting for BodyPose...")
    initial_msg = wait_for_body_pose(body_pose)
    root_pose, joint_rel_cum = initialize_from_body_pose(initial_msg)
    root_rel_cum = np.zeros(3, dtype=np.float32)

    print(f"Initial root pose: {root_pose}")
    for seconds_left in range(config.countdown_seconds, 0, -1):
        print(f"Replay starts in {seconds_left}s, press Ctrl+C to stop")
        sleep(1)

    sent_frames = 0
    try:
        for start in range(0, len(actions), config.chunk_size):
            action_chunk = actions[start : start + config.chunk_size]
            world_pose, root_rel_cum, joint_rel_cum, hand_action = (
                action_to_world_pose(
                    action_chunk,
                    root_pose,
                    root_rel_cum,
                    joint_rel_cum,
                )
            )

            print(
                f"Sending chunk {start // config.chunk_size + 1}, "
                f"frames={len(action_chunk)}"
            )

            for pose_frame, hand_frame in zip(world_pose, hand_action):
                xyz = pose_frame[:, :3]
                wxyz = normalize_quaternion(pose_frame[:, 3:7]).astype(np.float32)
                body_publisher.send_msg(xyz=xyz, wxyz=wxyz)
                hand_publisher.send(hand_frame)
                sent_frames += 1
                send_rate.sleep()

    except KeyboardInterrupt:
        print(f"Replay stopped: sent {sent_frames}/{len(actions)} frames")
    else:
        print(f"Replay finished: sent {sent_frames} frames")


if __name__ == "__main__":
    main(tyro.cli(ReplayConfig))

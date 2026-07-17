from dataclasses import dataclass, field

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tyro
from scipy.spatial.transform import Rotation as R

from data_res.log import get_logger
from data_res.dds import MocapConfig, MocapUE5G115MsgSubscriber

logger = get_logger(__name__)

matplotlib.rcParams["toolbar"] = "None"

_MOCAP_PARENTS = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 0, 12, 13]


@dataclass
class VisualizeMocapConfig:
    mocap: MocapConfig = field(default_factory=MocapConfig)
    """Mocap DDS subscriber configuration."""
    axis_radius: float = 1.0
    """XY visualization radius around the root joint."""
    z_min: float = 0.0
    """Minimum Z axis value."""
    z_max: float = 2.0
    """Maximum Z axis value."""
    frame_axis_scale: float = 0.1
    """Length of each joint orientation axis."""
    pause_s: float = 0.001
    """Matplotlib pause duration per frame."""
    print_metadata: bool = False
    """Print mocap fps and timestamp for each received frame."""


def plot_stickman(ax, stickman: np.ndarray, color: str, cfg: VisualizeMocapConfig) -> None:
    xyz = stickman[:, :3]
    x_vals, y_vals, z_vals = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    ax.scatter(x_vals, y_vals, z_vals, c=color, marker="o")

    for joint_idx, parent_idx in enumerate(_MOCAP_PARENTS):
        if parent_idx == -1:
            continue
        ax.plot(
            [xyz[joint_idx, 0], xyz[parent_idx, 0]],
            [xyz[joint_idx, 1], xyz[parent_idx, 1]],
            [xyz[joint_idx, 2], xyz[parent_idx, 2]],
            color=color,
            linewidth=1,
        )

    for joint_xyz, joint_wxyz in zip(xyz, stickman[:, 3:7]):
        quat_xyzw = joint_wxyz[[1, 2, 3, 0]]
        rot_matrix = R.from_quat(quat_xyzw).as_matrix()

        for axis_idx, axis_color in enumerate(("r", "g", "b")):
            axis = cfg.frame_axis_scale * rot_matrix[:, axis_idx]
            ax.plot(
                [joint_xyz[0], joint_xyz[0] + axis[0]],
                [joint_xyz[1], joint_xyz[1] + axis[1]],
                [joint_xyz[2], joint_xyz[2] + axis[2]],
                f"{axis_color}-",
                linewidth=2,
            )

    root = xyz[0]
    ax.set_xlim([root[0] - cfg.axis_radius, root[0] + cfg.axis_radius])
    ax.set_ylim([root[1] - cfg.axis_radius, root[1] + cfg.axis_radius])
    ax.set_zlim([cfg.z_min, cfg.z_max])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


def visualize_mocap(cfg: VisualizeMocapConfig) -> None:
    subscriber = MocapUE5G115MsgSubscriber(cfg.mocap)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    logger.info(f"visualizing mocap topic '{cfg.mocap.topic_name}'")

    while plt.fignum_exists(fig.number):
        msg = subscriber.get_msg()
        if msg is None:
            plt.pause(cfg.pause_s)
            continue

        if cfg.print_metadata:
            logger.info(f"fps={msg.fps}, timestamp={msg.timestamp}")

        stickman = subscriber.msg_to_data(msg)
        ax.clear()
        plot_stickman(ax, stickman, "black", cfg)
        plt.pause(cfg.pause_s)


if __name__ == "__main__":
    visualize_mocap(tyro.cli(VisualizeMocapConfig))

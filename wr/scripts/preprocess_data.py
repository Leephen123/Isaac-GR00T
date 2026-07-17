import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from tqdm import tqdm
import tyro

from data_res.log import get_logger
from data_res.transforms import compute_relative, quaternion_to_rotation_6d, compute_imu_relative
from data_res.utils import NumpyEncoder, load_mocap_imu_data
from data_res.dds import MOCAP_NUM_JOINTS, MOCAP_POS_DIM, MOCAP_QUAT_DIM, BODY_POSE_WXYZ_SIZE

logger = get_logger(__name__)


@dataclass
class PreprocessConfig:
    input_dirs: list[Path] = field(
        default_factory=lambda: [
            Path("/liujinxin/dataset/piper/G1/0514_clean_desk_place_sofa_g1_fast"),
            Path("/liujinxin/dataset/piper/G1/0515_clean_desk_place_sofa_g1_fast"),
            Path("/liujinxin/dataset/piper/G1/0518_clean_desk_place_sofa_g1_B"),
            Path("/liujinxin/dataset/piper/G1/0518_clean_desk_place_sofa_g1_fast"),
            Path("/liujinxin/dataset/piper/G1/0519_clean_desk_place_sofa_g1_B"),
            Path("/liujinxin/dataset/piper/G1/0520_clean_desk_place_sofa_g1_B"),
            Path("/liujinxin/dataset/piper/G1/0521_clean_desk_place_sofa_g1_B"),
        ]
    )
    """List of root directories, each containing episode_* subdirs with data.json files."""


def standardize_mocap(data: np.ndarray) -> np.ndarray:
    expected_joints = MOCAP_NUM_JOINTS
    expected_pose_dim = MOCAP_POS_DIM + MOCAP_QUAT_DIM
    if data.ndim != 3 or data.shape[1] != expected_joints or data.shape[2] != expected_pose_dim:
        raise ValueError(f"Expected mocap data shape (T, {expected_joints}, {expected_pose_dim}), got {data.shape}")

    num_frames, num_joints, pose_dim = data.shape
    reference = data[0, 0].copy()

    reference_tiled = np.tile(reference, (num_frames * num_joints, 1))
    data_flat = data.reshape(num_frames * num_joints, pose_dim)
    relative_flat = compute_relative(reference_tiled, data_flat)
    
    relative = relative_flat.reshape(num_frames, num_joints, pose_dim)
    return relative

def standardize_imu(data: np.ndarray) -> np.ndarray:
    expected_pose_dim = BODY_POSE_WXYZ_SIZE
    if data.ndim != 2 or data.shape[1] != expected_pose_dim:
        raise ValueError(f"Expected mocap data shape (T, {expected_pose_dim}), got {data.shape}")

    # remove the yaw angle to align with mocap data
    relative = compute_imu_relative(data, data)
    return relative

def preprocess_data(cfg: PreprocessConfig) -> None:
    for base_dir in cfg.input_dirs:
        episode_dirs = [episode_dir for episode_dir in sorted(base_dir.glob("episode_*")) if episode_dir.is_dir()]
        for episode_dir in tqdm(episode_dirs, desc=f"Processing {base_dir.name}", unit="episode"):
            data_json = episode_dir / "data.json"
            if not data_json.exists():
                raise RuntimeError(f"No data.json in {episode_dir}.")

            mocap_data, imu_data = load_mocap_imu_data(data_json)
            if mocap_data is None:
                raise RuntimeError(f"Failed to load mocap from {data_json}.")
            if imu_data is None:
                raise RuntimeError(f"Failed to load imu from {data_json}.")
                
            mocap_standard = standardize_mocap(mocap_data)
            imu_standard = standardize_imu(imu_data)

            mocap_standard_6d = quaternion_to_rotation_6d(mocap_standard)
            imu_standard_6d = quaternion_to_rotation_6d(imu_standard)

            with data_json.open("r", encoding="utf-8") as f:
                records = json.load(f)

            for i, step in enumerate(records):
                step["mocap"] = mocap_standard_6d[i].tolist()
                step["imu"] = imu_standard_6d[i].tolist()

            output_path = episode_dir / "data_standard_6D.json"
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(records, f, cls=NumpyEncoder, indent=2)
            logger.info(f"Saved: {output_path}")


if __name__ == "__main__":
    preprocess_data(tyro.cli(PreprocessConfig))

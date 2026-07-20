import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from tqdm import tqdm
import tyro

from data_res.log import get_logger
from data_res.transforms import quaternion_to_rotation_6d, mocap_to_root_relative_delta
from data_res.utils import NumpyEncoder, load_mocap_imu_data, standardize_mocap, standardize_imu

logger = get_logger(__name__)


@dataclass
class PreprocessConfig:
    input_dirs: list[Path] = field(
        default_factory=lambda: [
            Path("/liujinxin/dataset/piper/G1/0515_clean_desk_place_sofa_g1_fast"),
            Path("/liujinxin/dataset/piper/G1/0518_clean_desk_place_sofa_g1_B"),
            Path("/liujinxin/dataset/piper/G1/0518_clean_desk_place_sofa_g1_fast"),
            Path("/liujinxin/dataset/piper/G1/0519_clean_desk_place_sofa_g1_B"),
            Path("/liujinxin/dataset/piper/G1/0520_clean_desk_place_sofa_g1_B"),
            Path("/liujinxin/dataset/piper/G1/0521_clean_desk_place_sofa_g1_B"),
        ]
    )
    downsample_rate: int = 2
    """List of root directories, each containing episode_* subdirs with data.json files."""


def preprocess_data(cfg: PreprocessConfig) -> None:
    for base_dir in cfg.input_dirs:
        episode_dirs = [episode_dir for episode_dir in sorted(base_dir.glob("episode_*")) if episode_dir.is_dir()]
        for episode_dir in tqdm(episode_dirs, desc=f"Processing {base_dir.name}", unit="episode"):
            data_json = episode_dir / "data.json"
            if not data_json.exists():
                raise RuntimeError(f"No data.json in {episode_dir}.")

            mocap_data, imu_data = load_mocap_imu_data(data_json, downsample_rate = cfg.downsample_rate)
            if mocap_data is None:
                raise RuntimeError(f"Failed to load mocap from {data_json}.")
            if imu_data is None:
                raise RuntimeError(f"Failed to load imu from {data_json}.")
                
            mocap_standard = standardize_mocap(mocap_data)
            imu_standard = standardize_imu(imu_data)

            mocap_standard_6d = quaternion_to_rotation_6d(mocap_standard)
            imu_standard_6d = quaternion_to_rotation_6d(imu_standard)

            mocap_root_relative = mocap_to_root_relative_delta(mocap_standard_6d)

            with data_json.open("r", encoding="utf-8") as f:
                records = json.load(f)

            sampled_records = [
                step for index, step in enumerate(records)
                if index % cfg.downsample_rate == 0
            ]

            for i, step in enumerate(sampled_records):
                step["mocap"] = mocap_root_relative[i].tolist()
                step["imu"] = imu_standard_6d[i].tolist()

            output_path = episode_dir / "data_root_relative_6D.json"
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(sampled_records, f, cls=NumpyEncoder, indent=2)
            logger.info(f"Saved: {output_path}")


if __name__ == "__main__":
    preprocess_data(tyro.cli(PreprocessConfig))

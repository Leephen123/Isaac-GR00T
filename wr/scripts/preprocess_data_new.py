import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from tqdm import tqdm
import tyro
import sys
sys.path.append("/liujinxin/liyifan/gr00tN1.7/Isaac-GR00T/wr")
from data_res.log import get_logger
from data_res.transforms import quaternion_to_rotation_6d, mocap_to_root_relative_delta
from data_res.utils import NumpyEncoder, load_mocap_imu_data, standardize_mocap, standardize_imu

logger = get_logger(__name__)


@dataclass
class PreprocessConfig:
    input_dirs: list[Path] = field(
        default_factory=lambda: [
            # Path("/liujinxin/dataset/piper/G1/0526_clean_desk_place_sofa_g1_B"),
            # Path("/liujinxin/dataset/piper/G1/0527_clean_desk_place_sofa_g1_B"),
            # Path("/liujinxin/dataset/piper/G1/0528_clean_desk_place_sofa_g1_B"),
            # Path("/liujinxin/dataset/piper/G1/0529_clean_desk_place_sofa_g1_B"),
            # Path("/liujinxin/dataset/piper/G1/0601_clean_desk_place_sofa_g1_B"),

            # Path("/liujinxin/dataset/piper/G1/0626_pick_cube_bottle_g1"),
            # Path("/liujinxin/dataset/piper/G1/0629_pick_cube_bottle_g1"),
            
            # Path("/liujinxin/dataset/piper/G1/0710_pick_cube_bottle_g1_fix_2"),
            # Path("/liujinxin/dataset/piper/G1/0713_pick_cube_bottle_g1_fix_2"),

            # Path("/liujinxin/dataset/piper/G1/0717_clean_items_basket_2"),
            # Path("/liujinxin/dataset/piper/G1/0717_clean_items_basket_g1_2_2"),
            # Path("/liujinxin/dataset/piper/G1/0717_clean_items_basket_g1_mid_search_2"),
            # Path("/liujinxin/dataset/piper/G1/0717_clean_items_basket_g1_mid_walk_diagonally_2"),
            # Path("/liujinxin/dataset/piper/G1/0717_clean_items_basket_mistake_2"),

            # Path("/liujinxin/dataset/piper/G1/0720_clean_items_basket_g1_2"),
            # Path("/liujinxin/dataset/piper/G1/0720_clean_items_basket_g1_mistake_case12_1"),
            # Path("/liujinxin/dataset/piper/G1/0720_clean_items_basket_g1_mistake_case32_3"),
            # Path("/liujinxin/dataset/piper/G1/0720_clean_items_basket_g1_mistake_closecatch_1"),
            # Path("/liujinxin/dataset/piper/G1/0720_clean_items_basket_g1_mistake_closecatch_3"),

            Path("/liujinxin/dataset/piper/G1/0616_pick_cube_bottle_g1"),
            Path("/liujinxin/dataset/piper/G1/0617_pick_cube_bottle_g1_new1"),
            Path("/liujinxin/dataset/piper/G1/0623_pick_cube_bottle_four_g1"),
            Path("/liujinxin/dataset/piper/G1/0624_pick_cube_bottle_four_g1_1"),
            Path("/liujinxin/dataset/piper/G1/0624_pick_cube_bottle_four_g1_2"),
            Path("/liujinxin/dataset/piper/G1/0624_pick_cube_bottle_four_g1_3"),
            Path("/liujinxin/dataset/piper/G1/0625_pick_cube_bottle_four_g1"),
            Path("/liujinxin/dataset/piper/G1/0625_pick_cube_bottle_four_g1_2"),
            Path("/liujinxin/dataset/piper/G1/0625_pick_cube_bottle_four_g1_3"),
            Path("/liujinxin/dataset/piper/G1/0626_pick_cube_bottle_four_g1_5cube"),
            Path("/liujinxin/dataset/piper/G1/0626_pick_cube_bottle_g1_5cube"),
            Path("/liujinxin/dataset/piper/G1/0626_pick_cube_bottle_g1_mid"),

            # Path("/liujinxin/dataset/piper/G1/0629_pick_cube_bottle_g1"),
            # Path("/liujinxin/dataset/piper/G1/0626_pick_cube_bottle_g1"),
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

            output_path = episode_dir / "data_all_points_relative_6D.json"
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(sampled_records, f, cls=NumpyEncoder, indent=2)
            logger.info(f"Saved: {output_path}")


if __name__ == "__main__":
    preprocess_data(tyro.cli(PreprocessConfig))

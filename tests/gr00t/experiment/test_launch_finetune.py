import os

from gr00t.configs.finetune_config import FinetuneConfig
from gr00t.experiment.launch_finetune import build_dataset_configs
import pytest


def test_finetune_config_accepts_local_backbone_path():
    config = FinetuneConfig(
        base_model_path="model",
        backbone_model_path="/models/nvidia-Cosmos-Reason2-2B",
    )

    assert config.backbone_model_path == "/models/nvidia-Cosmos-Reason2-2B"


def test_build_dataset_configs_supports_legacy_path_group():
    config = FinetuneConfig(
        base_model_path="model",
        dataset_path=os.pathsep.join(["dataset-a", "dataset-b"]),
        embodiment_tag="UNITREE_G1_29DOF",
    )

    assert build_dataset_configs(config) == [
        {
            "dataset_paths": ["dataset-a", "dataset-b"],
            "mix_ratio": 1.0,
            "embodiment_tag": "unitree_g1_29dof",
        }
    ]


def test_build_dataset_configs_supports_independent_datasets():
    config = FinetuneConfig(
        base_model_path="model",
        dataset_paths=["dataset-a", "dataset-b"],
        dataset_mix_ratios=[1.0, 2.0],
        dataset_embodiment_tags=["UNITREE_G1_29DOF", "unitree_g1_29dof_hand"],
    )

    assert build_dataset_configs(config) == [
        {
            "dataset_paths": ["dataset-a"],
            "mix_ratio": 1.0,
            "embodiment_tag": "unitree_g1_29dof",
        },
        {
            "dataset_paths": ["dataset-b"],
            "mix_ratio": 2.0,
            "embodiment_tag": "unitree_g1_29dof_hand",
        },
    ]


@pytest.mark.parametrize(
    "config",
    [
        FinetuneConfig(base_model_path="model"),
        FinetuneConfig(
            base_model_path="model",
            dataset_path="dataset-a",
            dataset_paths=["dataset-b"],
            embodiment_tag="UNITREE_G1_29DOF",
        ),
    ],
)
def test_build_dataset_configs_requires_exactly_one_path_mode(config):
    with pytest.raises(ValueError, match="Exactly one"):
        build_dataset_configs(config)

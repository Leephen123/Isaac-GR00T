"""Validate mocap preprocessing and inference postprocessing without a model.

The script reproduces the current preprocessing protocol from a raw ``data.json``,
then feeds the processed action values directly into the same restoration function
used by realtime inference.  Both whole-episode and 50-step chunked restoration are
checked against the downsampled raw mocap XYZ values.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


WR_ROOT = Path(__file__).resolve().parents[1]
if str(WR_ROOT) not in sys.path:
    sys.path.insert(0, str(WR_ROOT))

from data_res.transforms import (  # noqa: E402
    compute_absolute,
    compute_relative,
    mocap_to_root_relative,
    quaternion_to_rotation_6d,
    restore_mocap_from_root_relative,
    rotation_6d_to_quaternion,
)


SELECT_11_INDICES = [0, 2, 3, 6, 7, 9, 10, 11, 12, 13, 14]


def standardize_mocap(data: np.ndarray) -> np.ndarray:
    """Apply the same first-root reference transform as data_res.utils."""
    num_frames, num_joints, pose_dim = data.shape
    reference = data[0, 0].copy()
    reference_tiled = np.tile(reference, (num_frames * num_joints, 1))
    return compute_relative(reference_tiled, data.reshape(-1, pose_dim)).reshape(data.shape)


def load_records(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        records = json.load(file)
    if not isinstance(records, list) or not records:
        raise ValueError(f"Expected a non-empty JSON list in {path}")
    return records


def load_raw_mocap(records: list[dict], downsample_rate: int) -> np.ndarray:
    mocap = np.asarray(
        [record["mocap"] for index, record in enumerate(records) if index % downsample_rate == 0],
        dtype=np.float64,
    )
    if mocap.ndim != 3 or mocap.shape[1:] != (15, 7):
        raise ValueError(f"Expected downsampled raw mocap shape (T, 15, 7), got {mocap.shape}")
    return mocap


def load_processed_mocap(records: list[dict]) -> np.ndarray:
    mocap = np.asarray([record["mocap"] for record in records], dtype=np.float64)
    if mocap.ndim != 2 or mocap.shape[1] != 138:
        raise ValueError(f"Expected processed mocap shape (T, 138), got {mocap.shape}")
    return mocap


def select_runtime_action(processed_mocap: np.ndarray) -> np.ndarray:
    root_delta = processed_mocap[:, :3]
    mocap_15x9 = processed_mocap[:, 3:].reshape(-1, 15, 9)
    mocap_11x9 = mocap_15x9[:, SELECT_11_INDICES]
    return np.concatenate([root_delta, mocap_11x9.reshape(len(mocap_11x9), -1)], axis=1)


def restore_in_chunks(
    action_102: np.ndarray,
    init_root_xyz: np.ndarray,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    restored_chunks = []
    root_cumulative = np.asarray(init_root_xyz, dtype=action_102.dtype).copy()

    for start in range(0, len(action_102), chunk_size):
        restored, root_cumulative = restore_mocap_from_root_relative(
            action_102[start : start + chunk_size],
            root_cumulative,
        )
        restored_chunks.append(restored)

    return np.concatenate(restored_chunks, axis=0), root_cumulative


def relative_to_world(restored_11x9: np.ndarray, reference_root_pose7: np.ndarray) -> np.ndarray:
    restored_11x7 = rotation_6d_to_quaternion(restored_11x9)
    references = np.broadcast_to(
        reference_root_pose7,
        (restored_11x7.shape[0], restored_11x7.shape[1], 7),
    ).reshape(-1, 7)
    return compute_absolute(references, restored_11x7.reshape(-1, 7)).reshape(
        restored_11x7.shape
    )


def xyz_error(reference: np.ndarray, reconstructed: np.ndarray) -> dict[str, object]:
    error = np.abs(reconstructed[..., :3] - reference[..., :3])
    return {
        "max_abs": float(error.max()),
        "mean_abs": float(error.mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "per_point_max_abs": error.max(axis=(0, 2)).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_json", type=Path)
    parser.add_argument("processed_json", type=Path)
    parser.add_argument("--downsample-rate", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.downsample_rate <= 0:
        raise ValueError("--downsample-rate must be positive")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")

    raw_records = load_records(args.raw_json)
    processed_records = load_records(args.processed_json)
    raw_mocap = load_raw_mocap(raw_records, args.downsample_rate)
    processed_mocap = load_processed_mocap(processed_records)
    if len(raw_mocap) != len(processed_mocap):
        raise ValueError(
            "Frame count mismatch after downsampling: "
            f"raw={len(raw_mocap)}, processed={len(processed_mocap)}"
        )

    standardized_pose7 = standardize_mocap(raw_mocap)
    standardized_pose9 = quaternion_to_rotation_6d(standardized_pose7)
    reproduced_processed = mocap_to_root_relative(standardized_pose9)
    preprocess_abs_error = np.abs(reproduced_processed - processed_mocap)

    # Realtime model_action_to_abs_action converts the server response to float32.
    action_102 = select_runtime_action(processed_mocap).astype(np.float32)
    reference_11x7 = raw_mocap[:, SELECT_11_INDICES]
    reference_standardized_11x9 = standardized_pose9[:, SELECT_11_INDICES]
    init_root_xyz = standardized_pose9[0, 0, :3].astype(np.float32)

    restored_whole, _ = restore_mocap_from_root_relative(
        action_102,
        init_root_xyz,
    )
    restored_chunked, final_root = restore_in_chunks(
        action_102,
        init_root_xyz,
        args.chunk_size,
    )

    world_whole = relative_to_world(restored_whole, raw_mocap[0, 0])
    world_chunked = relative_to_world(restored_chunked, raw_mocap[0, 0])

    runtime_reference = raw_mocap[0, 0].copy()
    runtime_reference[2] = 1.0
    world_with_runtime_z = relative_to_world(restored_chunked, runtime_reference)

    report = {
        "raw_frames": len(raw_records),
        "processed_frames": len(processed_records),
        "downsample_rate": args.downsample_rate,
        "chunk_size": args.chunk_size,
        "selected_point_indices": SELECT_11_INDICES,
        "preprocess_reproduction": {
            "max_abs": float(preprocess_abs_error.max()),
            "mean_abs": float(preprocess_abs_error.mean()),
        },
        "standardized_xyz_whole_episode": xyz_error(
            reference_standardized_11x9, restored_whole
        ),
        "standardized_xyz_chunked": xyz_error(
            reference_standardized_11x9, restored_chunked
        ),
        "world_xyz_whole_episode": xyz_error(reference_11x7, world_whole),
        "world_xyz_chunked": xyz_error(reference_11x7, world_chunked),
        "whole_vs_chunked_xyz": xyz_error(world_whole, world_chunked),
        "world_xyz_with_runtime_root_z_1": xyz_error(reference_11x7, world_with_runtime_z),
        "raw_initial_root_z": float(raw_mocap[0, 0, 2]),
        "runtime_root_z_offset": float(1.0 - raw_mocap[0, 0, 2]),
        "final_continuation_root_xyz": final_root.tolist(),
    }

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output_records = []
        for frame_index, (reference, reconstructed) in enumerate(
            zip(reference_11x7, world_chunked)
        ):
            output_records.append(
                {
                    "frame_index": frame_index,
                    "source_frame_index": frame_index * args.downsample_rate,
                    "selected_point_indices": SELECT_11_INDICES,
                    "reference_xyz": reference[:, :3].tolist(),
                    "reconstructed_xyz": reconstructed[:, :3].tolist(),
                    "abs_error_xyz": np.abs(reconstructed[:, :3] - reference[:, :3]).tolist(),
                }
            )
        with args.output.open("w", encoding="utf-8") as file:
            json.dump({"report": report, "frames": output_records}, file, indent=2)

    print(json.dumps(report, indent=2))
    if args.output is not None:
        print(f"Detailed frame comparison saved to: {args.output}")


if __name__ == "__main__":
    main()

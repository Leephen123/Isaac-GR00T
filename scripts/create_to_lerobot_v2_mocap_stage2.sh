#!/usr/bin/env bash
set -euo pipefail

echo "Checking ffmpeg..."
if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ffmpeg not found. Installing..."
    apt-get update
    apt-get install -y ffmpeg
else
    echo "ffmpeg already installed: $(which ffmpeg)"
    ffmpeg -version | head -n 1
fi
MODEL_NAME="2026_0721_all_points_rel_stage2"
MODALITY_NAME="modality_window_with_hand"
EMBODIMENT_TAG="UNITREE_G1_29DOF_HAND"

REPO_NAME="pick_cube_bottle_g1_0626_0629_all_points_rel"
DATASET_PATH="/liujinxin/liyifan/Isaac-GR00T/dataset/${REPO_NAME}"

###
REPO_NAME1="pick_cube_bottle_g1_0710_0713_fix_all_points_rel"
DATASET_PATH1="/liujinxin/liyifan/Isaac-GR00T/dataset/${REPO_NAME1}"
INPUT_PATH1="/liujinxin/dataset/piper/G1/0710_pick_cube_bottle_g1_fix_2"
INPUT_PATH2="/liujinxin/dataset/piper/G1/0713_pick_cube_bottle_g1_fix_2"

# cd /liujinxin/liyifan/gr00tN1.7/Isaac-GR00T
# source /liujinxin/conda3/bin/activate dreamzero

# python scripts/convert_to_lerobot_with_hand_new.py \
#     --output_dir "${DATASET_PATH1}" \
#     --input_dirs "${INPUT_PATH1}" "${INPUT_PATH2}"

# cp "/liujinxin/liyifan/Isaac-GR00T/scripts/${MODALITY_NAME}.json" "${DATASET_PATH1}/meta/modality.json"

# conda deactivate
# source .venv/bin/activate

# python /liujinxin/liyifan/gr00tN1.7/Isaac-GR00T/gr00t/data/stats.py \
#     --dataset-path "${DATASET_PATH1}" \
#     --embodiment-tag "${EMBODIMENT_TAG}"

# deactivate

###
REPO_NAME2="0721_clean_items_basket_g1_all_points_rel"
DATASET_PATH2="/liujinxin/liyifan/Isaac-GR00T/dataset/${REPO_NAME2}"

cd /liujinxin/liyifan/gr00tN1.7/Isaac-GR00T/
source /liujinxin/conda3/bin/activate dreamzero

python scripts/convert_to_lerobot_with_hand_new.py \
    --output_dir "${DATASET_PATH2}"

cp "/liujinxin/liyifan/Isaac-GR00T/scripts/${MODALITY_NAME}.json" "${DATASET_PATH2}/meta/modality.json"


cd /liujinxin/liyifan/gr00tN1.7/Isaac-GR00T
conda deactivate
source .venv/bin/activate

python /liujinxin/liyifan/gr00tN1.7/Isaac-GR00T/gr00t/data/stats.py \
    --dataset-path "${DATASET_PATH2}" \
    --embodiment-tag "${EMBODIMENT_TAG}"


NUM_GPUS=4
BATCH_PER_GPU=90
GLOBAL_BATCH_SIZE=$((NUM_GPUS * BATCH_PER_GPU))
export WANDB_API_KEY="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"
export WANDB_ENTITY="liyifansmx-westlake-university"

# echo "REPO_NAME=${REPO_NAME}"
# echo "DATASET_PATH=${DATASET_PATH}"
# echo "EMBODIMENT_TAG=${EMBODIMENT_TAG}"
# echo "NUM_GPUS=${NUM_GPUS}"
# echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}"
# echo "MODALITY_NAME=${MODALITY_NAME}"

# export CUDA_VISIBLE_DEVICES=0,1,2,3
# torchrun --nproc_per_node="${NUM_GPUS}" --master_port=29500 \
#     gr00t/experiment/launch_finetune.py \
#     --base-model-path /liujinxin/liyifan/gr00tN1.7/Isaac-GR00T/checkpoints/nvidia-GR00T-N1.7-3B \
#     --backbone_model_path /liujinxin/liyifan/gr00tN1.7/Isaac-GR00T/checkpoints/nvidia-Cosmos-Reason2-2B \
#     --dataset-path-groups "${DATASET_PATH},${DATASET_PATH1}" "${DATASET_PATH2}" \
#     --dataset_embodiment_tags "${EMBODIMENT_TAG}" "${EMBODIMENT_TAG}" \
#     --dataset_mix_ratios "1" "2" \
#     --num_gpus "${NUM_GPUS}" \
#     --output-dir "./checkpoints/${MODEL_NAME}" \
#     --save_total_limit 5 \
#     --save-steps 5000 \
#     --max-steps 50000 \
#     --warmup_ratio 0.05 \
#     --weight_decay 1e-5 \
#     --learning_rate 1e-4 \
#     --global_batch_size "${GLOBAL_BATCH_SIZE}" \
#     --dataloader_num_workers 6 \
#     --action_horizon 50 \
#     --state_history_length 50 \
#     --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
#     --shortest-image-edge 256 \
#     --crop-fraction 0.92 \
#     --random_rotation_angle 8 \
#     --use_wandb \
#     --tune_llm 
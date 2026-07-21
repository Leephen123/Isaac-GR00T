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
DATE="2026-07-20"
MODEL_NAME="${DATE}_pick_cube_bottle_g1_0616-0626_0629_0719fix2"
REPO_NAME="pick_cube_bottle_g1_0626"
OUTPUT_PATH="/liujinxin/liyifan/Isaac-GR00T/dataset/${REPO_NAME}"
DATASET_PATH_1="/liujinxin/liyifan/Isaac-GR00T/dataset/pick_cube_bottle_g1_0616-0623"
DATASET_PATH_2="/liujinxin/liyifan/Isaac-GR00T/dataset/pick_cube_bottle_g1_0624-0625"
DATASET_PATH_3="/liujinxin/liyifan/Isaac-GR00T/dataset/${REPO_NAME}"
DATASET_PATH_4="/liujinxin/liyifan/Isaac-GR00T/dataset/0629_pick_cube_bottle_g1"
DATASET_PATH_5="/liujinxin/liyifan/Isaac-GR00T/dataset/G1_real_6D_window_cont_rel_0518-0604"
DATASET_PATH_6="/liujinxin/liyifan/Isaac-GR00T/dataset/0710_pick_cube_bottle_g1_fix_2"
EMBODIMENT_TAG_1="UNITREE_G1_29DOF_HAND"
EMBODIMENT_TAG_3="UNITREE_G1_29DOF"
MODALITY_NAME="modality_window_with_hand"

NUM_GPUS=4
BATCH_PER_GPU=90
GLOBAL_BATCH_SIZE=$((NUM_GPUS * BATCH_PER_GPU))
export WANDB_API_KEY="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"
export WANDB_ENTITY="liyifansmx-westlake-university"
echo "DATE=${DATE}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "REPO_NAME=${REPO_NAME}"
echo "DATASET_PATH_1=${DATASET_PATH_1}"
echo "DATASET_PATH_2=${DATASET_PATH_2}"
echo "DATASET_PATH_3=${DATASET_PATH_3}"
echo "EMBODIMENT_TAG_1=${EMBODIMENT_TAG_1}"
echo "EMBODIMENT_TAG_3=${EMBODIMENT_TAG_3}"
echo "NUM_GPUS=${NUM_GPUS}"
echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}"

# cd /liujinxin/liyifan/Isaac-GR00T
# source /liujinxin/conda3/bin/activate dreamzero

# python scripts/convert_to_lerobot_new_with_hand.py \
#     --output_dir "${OUTPUT_PATH}"

# cp "/liujinxin/liyifan/Isaac-GR00T/scripts/${MODALITY_NAME}.json" "${DATASET_PATH_3}/meta/modality.json"

# conda deactivate
# source .venv/bin/activate

# python /liujinxin/liyifan/Isaac-GR00T/gr00t/data/stats.py \
#     --dataset-path "${DATASET_PATH_3}" \
#     --embodiment-tag "${EMBODIMENT_TAG_1}"

ulimit -n 1048576 || true

cd /liujinxin/liyifan/gr00tN1.7/Isaac-GR00T
source .venv/bin/activate

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node="${NUM_GPUS}" --master_port=29500 \
    gr00t/experiment/launch_finetune.py \
    --base-model-path /liujinxin/liyifan/gr00tN1.7/Isaac-GR00T/checkpoints/2026-06-29_G1_real_6D_window_cont_rel_0518-0604/checkpoint-80000 \
    --backbone_model_path /liujinxin/liyifan/gr00tN1.7/Isaac-GR00T/checkpoints/nvidia-Cosmos-Reason2-2B \
    --dataset-path-groups "${DATASET_PATH_1},${DATASET_PATH_2},${DATASET_PATH_3},${DATASET_PATH_4},${DATASET_PATH_6}" \
    --dataset_embodiment_tags "${EMBODIMENT_TAG_1}" \
    --dataset_mix_ratios "1" \
    --num_gpus "${NUM_GPUS}" \
    --output-dir "./checkpoints/${MODEL_NAME}" \
    --save_total_limit 5 \
    --save-steps 5000 \
    --max-steps 50000 \
    --warmup_ratio 0.05 \
    --weight_decay 1e-5 \
    --learning_rate 3e-5 \
    --global_batch_size "${GLOBAL_BATCH_SIZE}" \
    --dataloader_num_workers 6 \
    --body_action_dim 102 \
    --hand_action_dim 12 \
    --hand_loss_weight 0.1 \
    --action_horizon 50 \
    --state_history_length 50 \
    --use_wandb \
    --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --shortest-image-edge 256 \
    --crop-fraction 0.92 \
    --random_rotation_angle 3 \
    --tune_llm

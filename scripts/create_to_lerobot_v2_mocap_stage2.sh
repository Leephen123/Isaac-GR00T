#!/usr/bin/env bash
set -euo pipefail

# echo "Checking ffmpeg..."
# if ! command -v ffmpeg >/dev/null 2>&1; then
#     echo "ffmpeg not found. Installing..."
#     apt-get update
#     apt-get install -y ffmpeg
# else
#     echo "ffmpeg already installed: $(which ffmpeg)"
#     ffmpeg -version | head -n 1
# fi
MODEL_NAME1="2026_0723_finetune_stage2"

EMBODIMENT_TAG="UNITREE_G1_29DOF_HAND"


###

DATASET_PATH1="/liujinxin/liyifan/Isaac-GR00T/dataset/pick_cube_bottle_g1_0616_0617_0623_0624_0625_0626"

DATASET_PATH2="/liujinxin/liyifan/Isaac-GR00T/dataset/pick_cube_bottle_g1_0626_0629"

DATASET_PATH3="/liujinxin/liyifan/Isaac-GR00T/dataset/pick_cube_bottle_g1_0710_0713_fix"


### stage 2
NUM_GPUS=8
BATCH_PER_GPU=90
GLOBAL_BATCH_SIZE=$((NUM_GPUS * BATCH_PER_GPU))
export WANDB_API_KEY="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"
export WANDB_ENTITY="liyifansmx-westlake-university"

cd /liujinxin/liyifan/gr00tN1.7/Isaac-GR00T
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nproc_per_node="${NUM_GPUS}" --master_port=29500 \
    gr00t/experiment/launch_finetune.py \
    --base-model-path /liujinxin/liyifan/gr00tN1.7/Isaac-GR00T/checkpoints/2026-06-29_G1_real_6D_window_cont_rel_0518-0604/checkpoint-80000 \
    --backbone_model_path /liujinxin/liyifan/gr00tN1.7/Isaac-GR00T/checkpoints/nvidia-Cosmos-Reason2-2B \
    --dataset-path-groups "${DATASET_PATH1},${DATASET_PATH2},${DATASET_PATH3}" \
    --dataset_embodiment_tags "${EMBODIMENT_TAG}" \
    --dataset_mix_ratios "1" \
    --num_gpus "${NUM_GPUS}" \
    --output-dir "./checkpoints/${MODEL_NAME1}" \
    --save_total_limit 5 \
    --save-steps 5000 \
    --max-steps 50000 \
    --warmup_ratio 0.05 \
    --weight_decay 1e-5 \
    --learning_rate 3e-5 \
    --global_batch_size "${GLOBAL_BATCH_SIZE}" \
    --dataloader_num_workers 6 \
    --action_horizon 50 \
    --body_action_dim 102 \
    --hand_action_dim 12 \
    --state_history_length 50 \
    --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --shortest-image-edge 256 \
    --crop-fraction 0.92 \
    --random_rotation_angle 8 \
    --use_wandb \
    --tune_llm 


### stage 3
MODEL_NAME2="2026_0723_finetune_stage3"
DATASET_PATH4="/liujinxin/liyifan/Isaac-GR00T/dataset/0720-0722_clean_items_basket_g1"

NUM_GPUS=8
BATCH_PER_GPU=90
GLOBAL_BATCH_SIZE=$((NUM_GPUS * BATCH_PER_GPU))
export WANDB_API_KEY="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"
export WANDB_ENTITY="liyifansmx-westlake-university"


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nproc_per_node="${NUM_GPUS}" --master_port=29500 \
    gr00t/experiment/launch_finetune.py \
    --base-model-path ./checkpoints/${MODEL_NAME1}/checkpoint-50000 \
    --backbone_model_path /liujinxin/liyifan/gr00tN1.7/Isaac-GR00T/checkpoints/nvidia-Cosmos-Reason2-2B \
    --dataset-path-groups "${DATASET_PATH4}" \
    --dataset_embodiment_tags "${EMBODIMENT_TAG}" \
    --dataset_mix_ratios "1" \
    --num_gpus "${NUM_GPUS}" \
    --output-dir "./checkpoints/${MODEL_NAME2}" \
    --save_total_limit 5 \
    --save-steps 5000 \
    --max-steps 50000 \
    --warmup_ratio 0.05 \
    --weight_decay 1e-5 \
    --learning_rate 2e-5 \
    --global_batch_size "${GLOBAL_BATCH_SIZE}" \
    --dataloader_num_workers 6 \
    --action_horizon 50 \
    --body_action_dim 102 \
    --hand_action_dim 12 \
    --state_history_length 50 \
    --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
    --shortest-image-edge 256 \
    --crop-fraction 0.92 \
    --random_rotation_angle 8 \
    --use_wandb \
    --tune_llm 
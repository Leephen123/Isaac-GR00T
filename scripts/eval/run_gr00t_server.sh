#!/usr/bin/env bash

set -e

cd /liujinxin/liyifan/gr00tN1.7/Isaac-GR00T

source /liujinxin/liyifan/gr00tN1.7/Isaac-GR00T/.venv/bin/activate

export CUDA_VISIBLE_DEVICES=0

python /liujinxin/liyifan/gr00tN1.7/Isaac-GR00T/gr00t/eval/run_gr00t_server.py \
    --model_path /liujinxin/liyifan/gr00tN1.7/Isaac-GR00T/checkpoints/2026-07-20_pick_cube_bottle_g1_0616-0626_0629_0719fix2/checkpoint-5000 \
    --embodiment-tag UNITREE_G1_29DOF_HAND \
    --port 9002
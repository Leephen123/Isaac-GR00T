#!/usr/bin/env bash

set -e

cd /liujinxin/liyifan/gr00tN1.7/Isaac-GR00T

source /liujinxin/liyifan/gr00tN1.7/Isaac-GR00T/.venv/bin/activate

export CUDA_VISIBLE_DEVICES=0

python /liujinxin/liyifan/gr00tN1.7/Isaac-GR00T/gr00t/eval/run_gr00t_server.py \
    --model_path /liujinxin/liyifan/gr00tN1.7/Isaac-GR00T/checkpoints/2026-06-29_G1_real_6D_window_cont_rel_0518-0604/checkpoint-80000 \
    --embodiment-tag UNITREE_G1_29DOF \
    --port 9002
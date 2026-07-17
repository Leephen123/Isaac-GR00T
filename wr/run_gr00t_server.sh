#!/usr/bin/env bash

set -e

cd /liujinxin/liyifan/Isaac-GR00T

source /liujinxin/liyifan/Isaac-GR00T/.venv/bin/activate

export CUDA_VISIBLE_DEVICES=0

python /liujinxin/liyifan/Isaac-GR00T/gr00t/eval/run_gr00t_server.py \
    --model_path /liujinxin/zhaowei/Isaac-GR00T/logs_output/g1_29dof_finetune_real_6D_window_cont_rel/checkpoint-10000 \
    --embodiment-tag UNITREE_G1_29DOF \
    --port 9002
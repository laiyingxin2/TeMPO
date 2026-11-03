#!/usr/bin/env bash
# Evaluate TeMPO on LLaVA-Video-7B with the vendored lmms-eval.
# Run from the repository root:  bash scripts/eval_llava_video_tempo.sh
set -e

RATIO=${RETAIN_RATIO:-0.10}
TASKS=${TASKS:-videomme,egoschema}
OUT=${OUT:-./logs/llava_video_tempo_r${RATIO}}

WRAPPER=tempo RETAIN_RATIO=${RATIO} \
accelerate launch --num_processes 1 -m lmms_eval \
    --model llava_vid \
    --model_args pretrained=lmms-lab/LLaVA-Video-7B-Qwen2,conv_template=qwen_1_5,max_frames_num=64,mm_spatial_pool_mode=average \
    --tasks "${TASKS}" \
    --batch_size 1 \
    --log_samples \
    --output_path "${OUT}"

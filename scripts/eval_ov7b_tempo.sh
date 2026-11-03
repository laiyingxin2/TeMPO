#!/usr/bin/env bash
# Evaluate TeMPO on LLaVA-OneVision-7B with the vendored lmms-eval.
# Run from the repository root:  bash scripts/eval_ov7b_tempo.sh
set -e

RATIO=${RETAIN_RATIO:-0.10}
TASKS=${TASKS:-videomme,longvideobench_val_v,mlvu,egoschema}
OUT=${OUT:-./logs/ov7b_tempo_r${RATIO}}

WRAPPER=tempo RETAIN_RATIO=${RATIO} \
accelerate launch --num_processes 1 -m lmms_eval \
    --model llava_onevision \
    --model_args pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,model_name=llava_qwen,max_frames_num=32 \
    --tasks "${TASKS}" \
    --batch_size 1 \
    --log_samples \
    --output_path "${OUT}"

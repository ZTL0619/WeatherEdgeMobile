#!/usr/bin/env bash
set -euo pipefail

# Usage: CDD11_ROOT=/path/to/uncertainty bash scripts/run_pipeline.sh
# Teacher is genuine conditional DDPM; student is deterministic one-input RGB.
: "${CDD11_ROOT:?Set CDD11_ROOT=/path/to/uncertainty (containing train/, test/, splits/)}"
DEVICE="${DEVICE:-cuda:0}"
BATCH_TEACHER="${BATCH_TEACHER:-4}"
BATCH_STUDENT="${BATCH_STUDENT:-8}"
TEACHER_EPOCHS="${TEACHER_EPOCHS:-80}"
STUDENT_EPOCHS="${STUDENT_EPOCHS:-40}"
DDIM_STEPS="${DDIM_STEPS:-20}"

uv run weatherdiff profile --width 32 --size 256
uv run weatherdiff train-teacher --data-root "$CDD11_ROOT" \
  --device "$DEVICE" --epochs "$TEACHER_EPOCHS" --batch-size "$BATCH_TEACHER" \
  --sample-steps "$DDIM_STEPS" --val-every 5
uv run weatherdiff cache-teacher --data-root "$CDD11_ROOT" \
  --device "$DEVICE" --teacher outputs/teacher/best.pt \
  --sample-steps "$DDIM_STEPS"
uv run weatherdiff train-student --data-root "$CDD11_ROOT" \
  --device "$DEVICE" --epochs "$STUDENT_EPOCHS" --batch-size "$BATCH_STUDENT"
uv run weatherdiff evaluate --data-root "$CDD11_ROOT" --model student \
  --checkpoint outputs/student/best.pt --split test --device "$DEVICE" \
  --outdir outputs/eval_student
uv run --extra export weatherdiff export-onnx \
  --checkpoint outputs/student/best.pt --output outputs/mobile/student_256.onnx

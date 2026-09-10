#!/usr/bin/env bash
set -euo pipefail

: "${DATA_PATH:?Set DATA_PATH to a prepared Speculators dataset with cached hidden states}"
OUTPUT_DIR=${OUTPUT_DIR:-./output/dflash_qwen3_4b_b16_dpard_offline}
TARGET_MODEL=${TARGET_MODEL:-Qwen/Qwen3-4B}
NUM_TRAIN_GPUS=${NUM_TRAIN_GPUS:-2}
TRAIN_DATA_RATIO=${TRAIN_DATA_RATIO:-0.9999899063307494}

torchrun --standalone --nproc_per_node "${NUM_TRAIN_GPUS}" -m speculators.train \
  --verifier-name-or-path "${TARGET_MODEL}" \
  --speculator-type dflash \
  --data-path "${DATA_PATH}" \
  --save-path "${OUTPUT_DIR}/checkpoints" \
  --block-size 16 \
  --no-sample-from-anchor \
  --max-anchors 512 \
  --num-layers 3 \
  --target-layer-ids 1 17 33 \
  --draft-vocab-size 151936 \
  --loss-implementation fused \
  --loss-fn renyi_half \
  --per-position-loss-weight dpard \
  --dpard-alpha 0.5 \
  --dflash-loss-reduction sequence-mean-valid-anchor \
  --gradient-accumulation-steps 2 \
  --optimizer adamw \
  --lr 6e-4 \
  --weight-decay 0.01 \
  --scheduler-type linear \
  --scheduler-warmup-ratio 0.04 \
  --epochs 6 \
  --total-seq-len 8192 \
  --train-data-ratio "${TRAIN_DATA_RATIO}" \
  --noise-std 0.05 \
  --hidden-states-dtype bfloat16 \
  --num-workers 8 \
  --prefetch-factor 4 \
  --seed 42 \
  --fsdp-shard \
  --on-missing raise \
  --checkpoint-freq 1.0 \
  --no-resume-from-checkpoint \
  --log-freq 25 \
  "$@"

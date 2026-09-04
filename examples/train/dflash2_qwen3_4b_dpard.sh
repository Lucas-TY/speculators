#!/usr/bin/env bash
set -euo pipefail

: "${DATA_PATH:?Set DATA_PATH to prepared Qwen3-4B training data}"
: "${VLLM_ENDPOINT:?Set VLLM_ENDPOINT to the verifier OpenAI endpoint}"
OUTPUT_DIR=${OUTPUT_DIR:-./output/dflash2_qwen3_4b_dpard}
NUM_TRAIN_GPUS=${NUM_TRAIN_GPUS:-2}

torchrun --standalone --nproc_per_node "${NUM_TRAIN_GPUS}" -m speculators.train \
  --verifier-name-or-path Qwen/Qwen3-4B \
  --speculator-type dflash2 \
  --data-path "${DATA_PATH}" \
  --vllm-endpoint "${VLLM_ENDPOINT}" \
  --save-path "${OUTPUT_DIR}/checkpoints" \
  --block-size 8 \
  --max-anchors 512 \
  --num-layers 5 \
  --target-layer-ids 1 9 17 25 33 \
  --selector-rank 256 \
  --selector-top-k 16 \
  --selector-loss-alpha 1.0 \
  --loss-implementation fused \
  --loss-fn renyi_half \
  --per-position-loss-weight dpard \
  --dpard-alpha 0.5 \
  --optimizer adamw \
  --lr 6e-4 \
  --weight-decay 0.0 \
  --scheduler-type cosine \
  --scheduler-warmup-ratio 0.04 \
  --epochs 1 \
  --total-seq-len 8192 \
  --seed 42 \
  --fsdp-shard \
  --on-missing generate \
  --on-generate delete \
  --checkpoint-freq 0.1

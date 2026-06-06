#!/bin/bash
set -euo pipefail

cd /home/user

python data/generate_dialogues.py --out-dir data/local_dialogues

python method/nbf_llm.py train \
  --train-data data/local_dialogues/train.jsonl \
  --embedding-model data/downloads/all-MiniLM-L6-v2 \
  --cache tmp/train_embeddings.pt \
  --out-dir checkpoints/nbf \
  --ssm-epochs "${SSM_EPOCHS:-16}" \
  --nbf-epochs "${NBF_EPOCHS:-24}" \
  --batch-size 64

python method/nbf_llm.py predict \
  --model-path checkpoints/nbf/nbf_model.pt \
  --input-data data/local_dialogues/test_attacks.jsonl \
  --cache tmp/test_embeddings.pt \
  --threshold "${NBF_THRESHOLD:-0.001}" \
  --output tmp/outputs/nbf_attacks.json

python method/nbf_llm.py predict \
  --model-path checkpoints/nbf/nbf_model.pt \
  --input-data data/local_dialogues/benign.jsonl \
  --cache tmp/benign_embeddings.pt \
  --threshold "${NBF_THRESHOLD:-0.001}" \
  --output tmp/outputs/nbf_benign.json
# Run the paper's proposed method.
#
# This script trains/runs the paper's method using code in method/.
#
# Usage:
#   bash scripts/method.sh

set -e

cd /home/user

# TODO: run the paper's method
# TODO: call scripts/evaluate.sh to score

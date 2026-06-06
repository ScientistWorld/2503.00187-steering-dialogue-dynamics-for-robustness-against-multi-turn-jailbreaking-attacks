#!/bin/bash
set -euo pipefail

cd /home/user

# Evaluate the visible train slice from saved method outputs only.
python eval/train/evaluate.py \
  --output-dir tmp/outputs \
  --scores scoring/scores_train.json

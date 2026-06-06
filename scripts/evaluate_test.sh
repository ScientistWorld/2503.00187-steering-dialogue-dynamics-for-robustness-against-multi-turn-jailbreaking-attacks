#!/bin/bash
set -euo pipefail

cd /home/user

# Evaluate the held-out test slice from saved method outputs only. When the
# train scores exist, the test scorer requires exactly the same method entries.
python eval/test/evaluate.py \
  --output-dir tmp/outputs \
  --method-set-from scoring/scores_train.json \
  --scores scoring/scores_test.json

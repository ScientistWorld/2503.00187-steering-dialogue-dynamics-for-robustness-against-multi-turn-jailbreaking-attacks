#!/bin/bash
# Run every TorchScript guard in tmp/outputs/guards/ on the visible dev split and
# write scoring/scores_train.json. Never imports method/.
set -euo pipefail
cd "$(dirname "$0")/.."
python eval/train/evaluate.py --guard-dir tmp/outputs/guards --scores scoring/scores_train.json

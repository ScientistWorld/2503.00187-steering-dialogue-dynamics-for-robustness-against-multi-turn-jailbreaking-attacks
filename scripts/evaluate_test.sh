#!/bin/bash
# Run every TorchScript guard in tmp/outputs/guards/ on the hidden test inputs held in
# eval/test/data and write scoring/scores_test.json. Never imports method/, never retrains.
set -euo pipefail
cd "$(dirname "$0")/.."
python eval/test/evaluate.py --guard-dir tmp/outputs/guards --scores scoring/scores_test.json

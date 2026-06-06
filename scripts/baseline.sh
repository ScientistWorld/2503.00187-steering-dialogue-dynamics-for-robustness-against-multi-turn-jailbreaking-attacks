#!/bin/bash
set -euo pipefail

cd /home/user

[ -e data/local_dialogues/test_attacks.jsonl ] || python data/generate_dialogues.py --out-dir data/local_dialogues
python baseline/keyword_guard.py --input-data data/local_dialogues/test_attacks.jsonl --output tmp/outputs/baseline_attacks.json
python baseline/keyword_guard.py --input-data data/local_dialogues/benign.jsonl --output tmp/outputs/baseline_benign.json
# Run a baseline method from the paper.
#
# Baselines are simple reference methods the paper compares against. 
# Implement this only if applicable, i.e. the paper has baselines.
# You do not need to implement all baselines to begin — 
# just enough to validate that the evaluation pipeline works 
# and provide a reference point. You can decide to expand more baselines later.
# Each baseline must be actually run and scored through evaluate.sh
# so the numbers are verified, not self-reported.
# You can decide what this baseline script should do yourself.
#
# Usage:
#   bash scripts/baseline.sh <name>

set -e

cd /home/user

# TODO: run baseline

#!/bin/bash
set -euo pipefail

cd /home/user

# Run a lightweight lexical guard baseline with the same output schema as
# the NBF method, so scripts/evaluate.sh can score both without imports.
[ -e data/local_dialogues/test_attacks.jsonl ] || python data/generate_dialogues.py --out-dir data/local_dialogues
python baseline/keyword_guard.py --input-data data/local_dialogues/test_attacks.jsonl --output tmp/outputs/baseline_attacks.json
python baseline/keyword_guard.py --input-data data/local_dialogues/benign.jsonl --output tmp/outputs/baseline_benign.json

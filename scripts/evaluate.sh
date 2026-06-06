#!/bin/bash
set -euo pipefail

cd /home/user

# Evaluate saved method outputs only. This script intentionally does not
# import or call method/ so future methods can be scored with the same
# output schema.
python eval/evaluate_outputs.py \
  --attack-output tmp/outputs/nbf_attacks.json \
  --benign-output tmp/outputs/nbf_benign.json \
  --eta0-attack-output tmp/outputs/nbf_eta0_attacks.json \
  --eta0-benign-output tmp/outputs/nbf_eta0_benign.json \
  --baseline-output tmp/outputs/baseline_attacks.json \
  --scores scoring/scores.json

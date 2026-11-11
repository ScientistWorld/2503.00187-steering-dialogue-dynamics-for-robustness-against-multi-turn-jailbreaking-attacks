#!/bin/bash
set -euo pipefail
cd /home/user
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
trap 'python redteam/harness.py restore' EXIT
for variant in better_safe_transition_seed_13 better_safe_transition_seed_14 better_safe_transition_seed_15 better_no_invariance_200; do
  echo "AUDIT_BEGIN $variant $(date -u +%FT%TZ)"
  bash "redteam/$variant/run.sh"
  echo "AUDIT_END $variant $(date -u +%FT%TZ)"
done
python redteam/inspect_training.py
python redteam/summarize.py

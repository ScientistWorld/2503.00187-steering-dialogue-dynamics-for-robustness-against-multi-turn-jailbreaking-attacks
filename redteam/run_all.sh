#!/bin/bash
set -euo pipefail
cd /home/user
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
trap 'python redteam/harness.py restore' EXIT
variants=(
  reference_saved reference_seed_13 reference_seed_14 reference_seed_15
  better_training_200 better_training_96 weaker_training_1 no_invariance_variant
  constant_allow_control constant_block_control
  label_leak_shortcut generator_rule_shortcut enumeration_shortcut train_lookup_shortcut
  invented_answers_shortcut small_test_shortcut file_split_shortcut
  size_declaration_shortcut margin_reencoding_shortcut stale_artifact_shortcut
  baseline_replacement_shortcut method_omission_shortcut empty_outputs_shortcut
  impossible_negative_size_shortcut null_size_shortcut missing_pair_shortcut
  prewritten_score_shortcut
)
for variant in "${variants[@]}"; do
  echo "AUDIT_BEGIN $variant $(date -u +%FT%TZ)"
  if bash "redteam/$variant/run.sh"; then
    echo "AUDIT_END $variant $(date -u +%FT%TZ)"
  else
    echo "AUDIT_ERROR $variant $(date -u +%FT%TZ)"
  fi
done
python redteam/summarize.py

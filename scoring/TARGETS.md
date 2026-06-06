# Evaluation Targets

## Primary: open_source_multiturn_safety_tradeoff
Tests whether a guardrail blocks unsafe turns in multi-turn conversations while preserving benign interactions.
- **Primary metric**: `asr` (lower is better)
- **Benefit metrics**: `filter_recall`, `prompt_harmfulness_f1` (higher is better)
- **Constraint metrics**: `helpfulness_retention` (higher is better), `over_refusal_rate` and `guard_model_params_M` (lower is better)
- **Robustness diagnostics**: `max_attack_asr`, `min_attack_filter_recall`, `min_helpfulness_retention`, and `zero_buffer_*` metrics are computed from the saved output margins to catch family-specific failures and threshold overfitting.
- **Run**: `scripts/reproduce.sh`, then inspect `scoring/scores.json`

## Exploratory: heldout_attack_generalization
The workspace can still generate held-out attack-family outputs, but the corrected local result is too far from the paper's Table 5 values to serve as a default scored target. Future agents may inspect `tmp/outputs/nbf_heldout_*.json` and rerun `eval/evaluate_outputs.py --include-heldout-generalization` while developing, but official scoring currently targets the validated core trade-off above.

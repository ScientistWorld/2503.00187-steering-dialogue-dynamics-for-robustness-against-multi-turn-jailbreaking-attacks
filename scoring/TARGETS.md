# Evaluation Targets

## Primary: open_source_multiturn_safety_tradeoff
Tests whether a guardrail blocks unsafe turns in multi-turn conversations while preserving benign interactions.
- **Primary metric**: `asr` (lower is better)
- **Benefit metrics**: `filter_recall`, `prompt_harmfulness_f1` (higher is better)
- **Constraint metrics**: `helpfulness_retention` (higher is better), `over_refusal_rate` and `guard_model_params_M` (lower is better)
- **Robustness diagnostics**: `max_attack_asr`, `min_attack_filter_recall`, `min_helpfulness_retention`, and `zero_buffer_*` metrics are computed from the saved output margins to catch family-specific failures and threshold overfitting.
- **Run**: `scripts/reproduce.sh`, then inspect `scoring/scores.json`

## Secondary: heldout_attack_generalization
Tests whether a guardrail trained without selected attack families still filters those held-out families.
- **Primary metric**: `asr` (lower is better)
- **Benefit metrics**: `asr` and `zero_buffer_asr` (lower is better), `filter_recall`, `zero_buffer_filter_recall`, and `prompt_harmfulness_f1` (higher is better)
- **Constraint metrics**: `helpfulness_retention` (higher is better), `over_refusal_rate` and `guard_model_params_M` (lower is better)
- **Robustness diagnostics**: worst-family and zero-buffer metrics use the same definitions as the primary experiment.
- **Held-out axis**: training excludes ActorAttack and Opposite-day style trajectories while evaluation uses them.
- **Run**: `scripts/reproduce.sh`, then inspect `scoring/scores.json`; set `RUN_GENERALIZATION=0` only for a shorter smoke run that intentionally omits this target.

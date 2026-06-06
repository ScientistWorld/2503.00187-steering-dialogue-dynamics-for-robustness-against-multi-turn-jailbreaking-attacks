# Evaluation Targets

## Primary: open_source_multiturn_safety_tradeoff
Tests whether a guardrail blocks unsafe turns in multi-turn conversations while preserving benign interactions.
- **Primary metric**: `asr` (lower is better)
- **Benefit metrics**: `filter_recall`, `prompt_harmfulness_f1` (higher is better)
- **Constraint metrics**: `helpfulness_retention` (higher is better), `over_refusal_rate` and `guard_model_params_M` (lower is better)
- **Robustness diagnostics**: `max_attack_asr`, `min_attack_filter_recall`, `min_helpfulness_retention`, and `zero_buffer_*` metrics are computed from the saved output margins to catch family-specific failures and threshold overfitting.
- **Default scored guard**: `nbf_steering` uses the stronger paper threshold. The eta=0 threshold remains available as an optional diagnostic via `eval/evaluate_outputs.py --include-threshold-axis`, but is excluded from the default scored contract because the local substitute run did not match the paper's eta=0 ASR closely enough.
- **Run**: `scripts/reproduce.sh`, then inspect `scoring/scores.json`

## Secondary: heldout_attack_generalization
Tests whether a guardrail trained without selected attack families still filters those held-out families.
- **Primary metric**: `asr` (lower is better)
- **Benefit metrics**: `filter_recall`, `prompt_harmfulness_f1` (higher is better)
- **Constraint metrics**: `helpfulness_retention` (higher is better), `over_refusal_rate` and `guard_model_params_M` (lower is better)
- **Robustness diagnostics**: worst-family and zero-buffer metrics use the same definitions as the primary experiment.
- **Held-out axis**: training excludes ActorAttack and Opposite-day style trajectories while evaluation uses them.
- **Run**: set `RUN_GENERALIZATION=1` before `scripts/reproduce.sh`, then inspect `scoring/scores.json`

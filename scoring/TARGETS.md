# Evaluation Targets

## Primary: open_source_multiturn_safety_tradeoff
Tests whether a guardrail blocks unsafe turns in multi-turn conversations while preserving benign interactions.
- **Primary metric**: `asr` (lower is better)
- **Benefit metrics**: `filter_recall`, `prompt_harmfulness_f1` (higher is better)
- **Constraint metrics**: `helpfulness_retention` (higher is better), `over_refusal_rate` and `guard_model_params_M` (lower is better)
- **Threshold axis**: `nbf_steering` uses the stronger paper threshold, while `nbf_steering_eta_0` measures the weaker default training threshold.
- **Run**: `scripts/reproduce.sh`, then inspect `scoring/scores.json`

## Secondary: loss_ablation_eta_0
Tests the Table 6 claim that the safety-invariance loss improves the safety/helpfulness trade-off.
- **Primary metric**: `asr` (lower is better)
- **Benefit metrics**: `filter_recall`, `helpfulness_retention`
- **Constraint metrics**: `over_refusal_rate`, `guard_model_params_M`
- **Run**: set `RUN_ABLATIONS=1` before `scripts/reproduce.sh`

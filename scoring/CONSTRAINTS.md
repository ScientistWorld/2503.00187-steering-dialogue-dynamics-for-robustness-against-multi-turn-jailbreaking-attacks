# Scientist Constraints

## Data
Use the provided `data/local_dialogues` generator and its fixed train, attack-test, and benign-test splits. Do not add external labeled safety data or regenerate the test split with easier examples.

## Guard Size
The guardrail, including any embedding model it requires at inference, must remain no larger than the paper-scale guardrail budget of 115M parameters unless the larger size is explicitly penalized through `guard_model_params_M`.

## Evaluation
The evaluator must consume saved outputs from `tmp/outputs/` and must not import from `method/`. The test split must not be used for fitting thresholds, training classifiers, or selecting checkpoints.

## Compliance Checklist
- [ ] Uses only the provided local dialogue train split for training
- [ ] Does not train or tune on `test_attacks.jsonl` or `benign.jsonl`
- [ ] Reports `asr`, helpfulness/over-refusal constraints, and guard size
- [ ] `scripts/evaluate.sh` works without importing `method/`

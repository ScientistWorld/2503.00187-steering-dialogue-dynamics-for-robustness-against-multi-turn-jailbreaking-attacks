# Design Notes

## Checked
- Compared `scripts/evaluate_train.sh` and `scripts/evaluate_test.sh`; train calls `eval/train/evaluate.py` and writes `scoring/scores_train.json`, while test calls `eval/test/evaluate.py` and writes `scoring/scores_test.json`.
- Checked for train/test cross-imports with `rg`; only the two shell script entrypoints reference the split folders.
- Confirmed the split evaluators consume saved artifacts from `tmp/outputs/` rather than importing or rerunning `method/`.
- Confirmed no `scoring/scores_train.json` or `scoring/scores_test.json` stubs are present.
- Ran `bash -n` on both split evaluation scripts. `jq` was not installed, so JSON validity was reviewed by inspection rather than parser execution.

## Changed
- Added anti-shortcut per-family metrics to both train and test evaluators:
  - `max_attack_asr`
  - `min_attack_filter_recall`
  - `min_helpfulness_retention`
- Added those metrics to both experiments in `scoring/reference.json` so aggregate gains cannot hide collapse on one attack or benign subgroup.
- Added required `description` fields to every nonzero metric entry in `scoring/reference.json`.
- Shifted experiment weights from `0.65/0.35` to `0.60/0.40` to give the held-out attack-family axis more influence without overpowering the main safety/helpfulness tradeoff.
- Fixed a filename-gaming issue in guard-size scoring: only the exact existing `baseline` stem gets the zero-parameter fallback, and future artifacts may declare `guard_model_params_M` directly. Invalid negative or nonnumeric declarations now fail evaluation.

## Runner Watchpoints
- The current method artifacts do not declare `guard_model_params_M`, so existing outputs still use evaluator fallbacks. Future methods should prefer a top-level `guard_model_params_M` field in both attack and benign JSON outputs.
- The split is an even/odd dialogue partition within each `(source_split, attack)` group. This is useful for catching ID-specific tuning, but a runner should watch for saturation if the local generated dataset is too easy after future method changes.
- Train/test evaluation reads the same saved full-output files and slices them internally. This is lightweight, but it means final methods must emit decisions for all dialogues, not only the visible train slice.

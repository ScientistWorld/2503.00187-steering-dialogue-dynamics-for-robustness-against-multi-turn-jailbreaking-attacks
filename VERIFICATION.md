# Verification Report (iteration 11)

## Checked
- `python validate.py --post-design` passes (only warning: `eval/test/compare.py` is a maintainer bootstrap tool that reads decision dumps, not a scorer).
- `scripts/evaluate_test.sh` takes no arguments, scores every `tmp/outputs/guards/*.pt`, overwrites a prewritten `scores_test.json` (probe: `asr -5` -> 0.1646), and writes worst values for failed guards in every metric of both experiments.
- `scoring/scores_test.json` and `scoring/scores_train.json` hold only the reference runs (`nbf_steering`, `nbf_steering_eta0`, `single_turn_guard`) from job `62478226-05a`, scored by the final evaluators.
- `eval/` never imports `method/`; `eval/train/` and `eval/test/` share no code; test inputs, labels and the test-data builders live only in `eval/test/`.
- `scripts/download.sh` fetches the authors' release archive (sha256-checked, only Circuit Breakers members converted, archive deleted) and the pinned all-mpnet-base-v2 revision; nothing references `shared/`.

## Final judgment
- Milestone: core_claim_plus
- Ready for gym use: yes
- Key limitations: offline replay of GPT-3.5 dialogues; helpfulness/over-refusal are guard pass rates; the L_SI ablation does not reproduce the paper's direction; training on the public HarmBench release cannot be detected by the evaluator (see HARDENING.md).

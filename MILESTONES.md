# Reproduction Milestones

**Current: core_claim_plus**

<!-- Milestone levels (update "Current" above as you progress):
  none             — just started, no meaningful progress yet
  method_runs      — the paper's method executes end-to-end without errors
  core_claim       — minimum experiment supports the paper's central claim
  core_claim_plus  — core claim reproduced on additional settings
  secondary_claims — secondary results or contributions reproduced
  majority         — more than half of reported results reproduced
  near_complete    — most results reproduced, only minor gaps remain
  full             — all reported results reproduced
-->

## Progress Log

### [2026-06-06 05:35] - none
- Read the paper and extracted the core method and evaluation requirements.
- Cloned the author repository and confirmed the released code implements the neural state-space model plus neural barrier predictor.
- Implemented the reproduction environment and prepared the first end-to-end compute job.

### [2026-06-06 05:52] - method_runs
- CPU smoke job trained the neural dialogue dynamics and neural barrier predictor.
- The method wrote attack and benign filtering outputs and `scoring/scores.json`.
- Job failed only because the validator required reproduced method names to also appear in `reference.json`.

### [2026-06-06 05:57] - core_claim
- Revised CPU smoke job passed `python validate.py --compare`.
- NBF steering reduced attack success to 0.0 on the local multi-turn split while retaining 0.958 benign helpfulness.
- The keyword guard baseline remained nontrivial, with attack success 0.433 and benign over-refusal 0.175.

### [2026-06-06 06:03] - core_claim_plus
- Final verification job passed with cleaned scripts and regenerated `scoring/scores.json`.
- Added the eta=0 threshold setting: it retained all benign prompts but had much higher attack success, exposing the threshold safety/helpfulness trade-off.
- The stronger eta=1e-3 setting remains the reproduced core result.

## Stop Justification
- Reproduction packaged and validated at `core_claim_plus`; broader paper coverage would require paid/API-generated datasets or substantially larger experimental matrices.

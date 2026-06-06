# Verification Report

## Initial Assessment
The workspace entered this final check with a prior `secondary_claims` result, but a follow-up patch changed the NBF steering semantics so filtered queries no longer advance the latent dialogue state. A CPU rerun has now regenerated the scored artifacts under the corrected semantics.

## Issues Found
- **Critical**: The previous scored artifacts were stale relative to the patched method. The rerun fixed this, but the corrected held-out result no longer supports the paper's secondary generalization claim.
- **Critical**: The corrected held-out experiment is too far from the paper's reported ASR/filter-recall values to pass paper-consistency checks, so it cannot remain in default scored artifacts while closing at `core_claim`.
- **Minor**: Root-level tracked `trajectory/*.jsonl` artifacts were deleted in the worktree but not committed. These raw trajectory files do not belong in the workspace root.
- **Minor**: `README.md` and `manifest.json` appear as modified character-device files in the worktree. They are unrelated root/system artifacts, so I did not alter them.

## Fixes Applied
- Committed removal of the tracked root `trajectory/*.jsonl` artifacts.
- CPU job `e7898f38-d24` regenerated method outputs, baseline outputs, held-out generalization outputs, train/test scores, and validation artifacts with the corrected filtered-state behavior.
- Removed the failed held-out generalization run from default `scores.json`, `scores_train.json`, and `scores_test.json`; it remains available as an explicitly requested exploratory diagnostic.
- Verified `python validate.py --compare` passes on the refreshed artifacts.
- Verified the HuggingFace model endpoint used by `scripts/download.sh` resolves successfully with `curl -I -L --max-time 15`.

## Final Judgment
The workspace is ready as a `core_claim` gym. The main safety/helpfulness trade-off is reproduced on the local substitute benchmark, evaluation is reusable, method-agnostic files are clean enough for from-scratch agents, and validation passes. The previous `secondary_claims` milestone is not justified after the corrected rerun because held-out attack generalization is weak.

- Milestone: core_claim
- Ready for gym use: yes
- Confidence: medium
- Key limitation: benchmark uses local synthetic substitute data, and the corrected held-out generalization experiment is exploratory rather than a reproduced secondary claim.

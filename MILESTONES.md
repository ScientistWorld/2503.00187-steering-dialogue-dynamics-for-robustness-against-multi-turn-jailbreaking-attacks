# Reproduction Milestones

**Current: core_claim_plus**

## Progress Log

### [2026-06-06 06:22] - core_claim_plus
- Existing reproduction already runs the paper's NBF dialogue dynamics and barrier filtering end-to-end on the local benchmark.
- Existing scores cover the main eta=1e-3 safety/helpfulness trade-off and an eta=0 threshold setting, giving additional coverage beyond the minimum core claim.
- Added a held-out attack-family generalization experiment modeled on Table 5; next submitted job will determine whether it supports `secondary_claims`.

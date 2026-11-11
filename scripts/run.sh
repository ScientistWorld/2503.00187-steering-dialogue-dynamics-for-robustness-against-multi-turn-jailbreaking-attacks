#!/bin/bash
# End-to-end reference reproduction (GPU job, no internet; run scripts/download.sh beforehand):
# train the paper's neural dialogue dynamics + neural barrier function guard (eta = 1e-3 and
# eta = 0) and the single-turn baseline, then score every guard in tmp/outputs/guards/ on the
# visible dev slice (scoring/scores_train.json) and the hidden test (scoring/scores_test.json).
# The hardening audit is hardening/run_hardening.sh (rank re-scoring: hardening/rescore_rank.sh).
set -euo pipefail
cd /home/user
bash scripts/reproduce.sh
bash scripts/evaluate_test.sh

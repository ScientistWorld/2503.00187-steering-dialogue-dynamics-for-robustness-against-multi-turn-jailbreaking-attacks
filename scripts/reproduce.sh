#!/bin/bash
set -euo pipefail

cd /home/user

# The packaged reproduction covers the primary trade-off plus the held-out
# attack-family generalization target. Set RUN_GENERALIZATION=0 only for a
# shorter smoke run.
export RUN_GENERALIZATION="${RUN_GENERALIZATION:-1}"

scripts/method.sh
scripts/baseline.sh
scripts/evaluate.sh

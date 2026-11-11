#!/bin/bash
# Train the reference guard and the baseline, then score them on the visible dev slice.
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/method.sh
bash scripts/baseline.sh
bash scripts/evaluate_train.sh

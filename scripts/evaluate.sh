#!/bin/bash
# Score every guard in tmp/outputs/guards/ on the visible dev slice (same as evaluate_train.sh).
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/evaluate_train.sh

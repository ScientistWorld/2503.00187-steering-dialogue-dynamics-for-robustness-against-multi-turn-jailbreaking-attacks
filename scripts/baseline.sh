#!/bin/bash
# Single-turn guard baseline: a context-free query classifier of the same head size.
set -euo pipefail
cd "$(dirname "$0")/.."
python baseline/single_turn_guard.py --epochs "${BASELINE_EPOCHS:-200}" --out tmp/outputs/guards/single_turn_guard.pt

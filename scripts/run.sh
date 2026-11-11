#!/bin/bash
# Job entry point (iteration 11 hardening check, paper scale). Afterwards this file is reset to
# the end-to-end reference reproduction: reproduce -> evaluate_train -> evaluate_test.
set -euo pipefail
cd /home/user
SMOKE=0 bash hardening/run_hardening.sh
python validate.py --post-design || true

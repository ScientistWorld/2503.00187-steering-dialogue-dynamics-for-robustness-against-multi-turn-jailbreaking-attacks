#!/bin/bash
# Job entry point (iteration 11 hardening check). Afterwards this file is reset to the
# end-to-end reference reproduction: reproduce -> evaluate_train -> evaluate_test.
set -euo pipefail
cd /home/user
SMOKE="${SMOKE:-1}" bash hardening/run_hardening.sh
python validate.py || true

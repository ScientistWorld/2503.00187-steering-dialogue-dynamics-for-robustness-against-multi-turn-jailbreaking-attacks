#!/bin/bash
# Job entry point (iteration 11: rescore the hardening panel under the final coefficients,
# CPU only). Afterwards this file is reset to the end-to-end reference reproduction.
set -euo pipefail
cd /home/user
bash hardening/rescore_rank.sh
python validate.py --post-design || true

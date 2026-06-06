#!/bin/bash
set -euo pipefail

cd /home/user
export TOKENIZERS_PARALLELISM=false
export SSM_EPOCHS="${SSM_EPOCHS:-4}"
export NBF_EPOCHS="${NBF_EPOCHS:-6}"
export RUN_ABLATIONS="${RUN_ABLATIONS:-1}"
scripts/reproduce.sh
python validate.py --compare

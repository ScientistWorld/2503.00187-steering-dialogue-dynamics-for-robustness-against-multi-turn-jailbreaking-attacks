#!/bin/bash
set -euo pipefail

cd /home/user
export TOKENIZERS_PARALLELISM=false
export SSM_EPOCHS="${SSM_EPOCHS:-8}"
export NBF_EPOCHS="${NBF_EPOCHS:-12}"
export RUN_GENERALIZATION="${RUN_GENERALIZATION:-1}"
export RUN_ABLATIONS="${RUN_ABLATIONS:-0}"
scripts/reproduce.sh
python validate.py --compare

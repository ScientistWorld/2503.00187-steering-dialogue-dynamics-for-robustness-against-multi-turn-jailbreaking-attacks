#!/bin/bash
# Train the paper's neural dialogue dynamics + neural barrier function on the
# training dialogues and export the steering guard at eta = 1e-3 (paper default)
# and eta = 0 into tmp/outputs/guards/.
set -euo pipefail
cd "$(dirname "$0")/.."
SEED="${NBF_SEED:-0}"
CK=checkpoints/nbf
mkdir -p "$CK" tmp/outputs/guards
python method/nbf_llm.py train-ssm --seed "$SEED" --epochs "${SSM_EPOCHS:-200}" --out "$CK/ssm_seed$SEED.pt"
python method/nbf_llm.py train-nbf --seed "$SEED" --epochs "${NBF_EPOCHS:-200}" \
  --ssm "$CK/ssm_seed$SEED.pt" --out "$CK/nbf_seed$SEED.pt"
python method/nbf_llm.py export --model "$CK/nbf_seed$SEED.pt" --eta 0.001 --out tmp/outputs/guards/nbf_steering.pt
python method/nbf_llm.py export --model "$CK/nbf_seed$SEED.pt" --eta 0 --out tmp/outputs/guards/nbf_steering_eta0.pt

#!/bin/bash
set -euo pipefail

cd /home/user

# Train the paper's neural dialogue dynamics and neural barrier predictor,
# then export filtering decisions at the default and eta=0 thresholds.
python data/generate_dialogues.py --out-dir data/local_dialogues

python method/nbf_llm.py train \
  --train-data data/local_dialogues/train.jsonl \
  --embedding-model data/downloads/all-MiniLM-L6-v2 \
  --cache tmp/train_embeddings.pt \
  --out-dir checkpoints/nbf \
  --ssm-epochs "${SSM_EPOCHS:-16}" \
  --nbf-epochs "${NBF_EPOCHS:-24}" \
  --batch-size 64

python method/nbf_llm.py predict \
  --model-path checkpoints/nbf/nbf_model.pt \
  --input-data data/local_dialogues/test_attacks.jsonl \
  --cache tmp/test_embeddings.pt \
  --threshold "${NBF_THRESHOLD:-0.001}" \
  --output tmp/outputs/nbf_attacks.json

python method/nbf_llm.py predict \
  --model-path checkpoints/nbf/nbf_model.pt \
  --input-data data/local_dialogues/benign.jsonl \
  --cache tmp/benign_embeddings.pt \
  --threshold "${NBF_THRESHOLD:-0.001}" \
  --output tmp/outputs/nbf_benign.json

python method/nbf_llm.py predict \
  --model-path checkpoints/nbf/nbf_model.pt \
  --input-data data/local_dialogues/test_attacks.jsonl \
  --cache tmp/test_embeddings.pt \
  --threshold 0 \
  --output tmp/outputs/nbf_eta0_attacks.json

python method/nbf_llm.py predict \
  --model-path checkpoints/nbf/nbf_model.pt \
  --input-data data/local_dialogues/benign.jsonl \
  --cache tmp/benign_embeddings.pt \
  --threshold 0 \
  --output tmp/outputs/nbf_eta0_benign.json

if [ "${RUN_GENERALIZATION:-0}" = "1" ]; then
  # Table 5-style generalization: train without ActorAttack and Opposite-day,
  # then evaluate the same NBF steering on those held-out attack families.
  python data/generate_dialogues.py \
    --out-dir data/local_dialogues_heldout \
    --train-attacks crescendo,acronym \
    --test-attacks actor,opposite \
    --benign-attacks actor,opposite

  python method/nbf_llm.py train \
    --train-data data/local_dialogues_heldout/train.jsonl \
    --embedding-model data/downloads/all-MiniLM-L6-v2 \
    --cache tmp/train_heldout_embeddings.pt \
    --out-dir checkpoints/nbf_heldout \
    --ssm-epochs "${SSM_EPOCHS:-16}" \
    --nbf-epochs "${NBF_EPOCHS:-24}" \
    --batch-size 64

  python method/nbf_llm.py predict \
    --model-path checkpoints/nbf_heldout/nbf_model.pt \
    --input-data data/local_dialogues_heldout/test_attacks.jsonl \
    --cache tmp/test_heldout_embeddings.pt \
    --threshold "${NBF_THRESHOLD:-0.001}" \
    --output tmp/outputs/nbf_heldout_attacks.json

  python method/nbf_llm.py predict \
    --model-path checkpoints/nbf_heldout/nbf_model.pt \
    --input-data data/local_dialogues_heldout/benign.jsonl \
    --cache tmp/benign_heldout_embeddings.pt \
    --threshold "${NBF_THRESHOLD:-0.001}" \
    --output tmp/outputs/nbf_heldout_benign.json
fi

if [ "${RUN_ABLATIONS:-0}" = "1" ]; then
  # Table 6 ablation: train the same NBF without the safety-invariance loss
  # L_SI, then evaluate at eta=0.
  python method/nbf_llm.py train \
    --train-data data/local_dialogues/train.jsonl \
    --embedding-model data/downloads/all-MiniLM-L6-v2 \
    --cache tmp/train_embeddings.pt \
    --out-dir checkpoints/nbf_no_si \
    --ssm-epochs "${SSM_EPOCHS:-16}" \
    --nbf-epochs "${NBF_EPOCHS:-24}" \
    --batch-size 64 \
    --invariance-weight 0.0

  python method/nbf_llm.py predict \
    --model-path checkpoints/nbf_no_si/nbf_model.pt \
    --input-data data/local_dialogues/test_attacks.jsonl \
    --cache tmp/test_embeddings.pt \
    --threshold 0 \
    --output tmp/outputs/nbf_no_si_attacks.json

  python method/nbf_llm.py predict \
    --model-path checkpoints/nbf_no_si/nbf_model.pt \
    --input-data data/local_dialogues/benign.jsonl \
    --cache tmp/benign_embeddings.pt \
    --threshold 0 \
    --output tmp/outputs/nbf_no_si_benign.json
fi

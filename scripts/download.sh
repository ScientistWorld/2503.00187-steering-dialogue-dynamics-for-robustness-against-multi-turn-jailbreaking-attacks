#!/bin/bash
# Fetch every external artifact the gym uses (run on a node with internet).
# Estimated download: ~190 MB archive (deleted after conversion) + ~440 MB encoder.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/downloads

# 1. The NBF authors' released multi-turn jailbreak dialogue embeddings
#    (README of github.com/HanjiangHu/NBF-LLM -> Google Drive train_val_embedding_data.zip).
#    Only the Circuit Breakers (training) members are converted into data/downloads/nbf_cb;
#    the archive itself is deleted afterwards.
if [ ! -s data/downloads/nbf_cb/dialogues.json ]; then
  tmpdir="$(mktemp -d)"
  curl -sfL "https://drive.usercontent.google.com/download?id=1EO8GKpuxQm-EZQn9i_00fJWg7yXXGWSF&export=download&confirm=t" \
    -o "$tmpdir/train_val_embedding_data.zip"
  echo "6593fba9ffe05df661853b14d6d51a8a743bf2a50738fa678165817206725808  $tmpdir/train_val_embedding_data.zip" | sha256sum -c -
  python3 data/prepare_release.py --zip "$tmpdir/train_val_embedding_data.zip" --out-dir data/downloads/nbf_cb
  rm -rf "$tmpdir"
fi

# 2. The paper's default sentence encoder (all-mpnet-base-v2). Guards consume
#    embeddings, so it is only needed to rebuild data/benign/*_u.f16.
ENC=data/downloads/all-mpnet-base-v2
REV=e8c3b32edf5434bc2275fc9bab85f82640a19130
if [ ! -s "$ENC/model.safetensors" ]; then
  mkdir -p "$ENC/1_Pooling"
  for f in 1_Pooling/config.json config.json config_sentence_transformers.json data_config.json \
           model.safetensors modules.json sentence_bert_config.json special_tokens_map.json \
           tokenizer.json tokenizer_config.json vocab.txt README.md; do
    curl -sfL "https://huggingface.co/sentence-transformers/all-mpnet-base-v2/resolve/$REV/$f" -o "$ENC/$f"
  done
fi

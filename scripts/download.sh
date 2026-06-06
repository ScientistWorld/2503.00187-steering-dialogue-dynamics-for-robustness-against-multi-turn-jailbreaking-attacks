#!/bin/bash
set -euo pipefail

# Estimated download: ~90 MB, disk usage ~100 MB.
# Downloads the local sentence embedding model used by the NBF reproduction.

mkdir -p data/downloads

if [ ! -e data/downloads/all-MiniLM-L6-v2 ]; then
  huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 \
    --local-dir data/downloads/all-MiniLM-L6-v2 \
    --local-dir-use-symlinks False
fi

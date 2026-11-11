#!/usr/bin/env python3
"""Embed the benign/harmful prompt splits with the paper's default sentence encoder.

Writes one float16 [n_turns, 768] file next to each JSONL (``<name>_u.f16``),
turns in file order. The encoder (all-mpnet-base-v2, unit-normalised output) is
the same one the authors used for the released attack-dialogue embeddings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", default="data/downloads/all-mpnet-base-v2")
    parser.add_argument(
        "--jsonl",
        nargs="+",
        default=["data/benign/benign_train.jsonl", "data/benign/benign_dev.jsonl", "eval/test/data/benign_test.jsonl"],
    )
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(args.encoder, device=device)
    for path in map(Path, args.jsonl):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        texts = [t for r in rows for t in r["turns"]]
        emb = model.encode(texts, batch_size=64, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
        emb = emb.float().cpu().to(torch.float16).contiguous()
        out = path.with_name(path.stem + "_u.f16")
        out.write_bytes(emb.numpy().tobytes())
        print(f"{path}: {len(rows)} items, {len(texts)} turns -> {out}")


if __name__ == "__main__":
    main()

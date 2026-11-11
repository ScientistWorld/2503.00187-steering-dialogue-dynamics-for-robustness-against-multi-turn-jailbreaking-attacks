#!/usr/bin/env python3
"""Mark hidden-test turns whose query embedding also occurs in the visible data.

The hidden attack dialogues are the authors' public HarmBench release and the
benign prompts come from public benchmarks, so a guard could store hidden-test
embeddings verbatim (a lookup table of the test set). ``eval/test/evaluate.py``
rejects a guard whose tensors contain near-copies of more than a small share
of the hidden-test queries. Generic turns that also occur in the visible data
("Can you elaborate?") are legitimately present in any memory of the training
set, so they are excluded from that count. This script records them:

    eval/test/data/visible_overlap.json
        {"cos": 0.995, "attack": [turn indices], "benign": [turn indices]}

Indices refer to rows of hb_u.f16 and benign_test_u.f16. Visible data: every
turn of data/downloads/nbf_cb (train + dev) and data/benign (train + dev).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
TEST = ROOT / "eval" / "test" / "data"
DIM = 768
COS = 0.995


def f16(path: Path) -> torch.Tensor:
    t = torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.float16)
    return t.view(-1, DIM).float()


def unit(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-8)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    visible = [ROOT / "data/downloads/nbf_cb/u.f16", ROOT / "data/benign/benign_train_u.f16",
               ROOT / "data/benign/benign_dev_u.f16"]
    for p in visible:
        if not p.exists():
            sys.exit(f"missing visible data {p}")
    keys = unit(torch.cat([f16(p) for p in visible])).to(device)
    out = {"cos": COS}
    for name, path in (("attack", TEST / "hb_u.f16"), ("benign", TEST / "benign_test_u.f16")):
        q = unit(f16(path)).to(device)
        best = torch.cat([(q[i : i + 1024] @ keys.t()).abs().max(dim=1).values for i in range(0, len(q), 1024)])
        idx = torch.nonzero(best >= COS).flatten().tolist()
        out[name] = idx
        out[f"n_{name}_turns"] = len(q)
        print(f"{name}: {len(idx)} / {len(q)} test turns have a visible near-duplicate (|cos| >= {COS})")
    (TEST / "visible_overlap.json").write_text(json.dumps(out) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

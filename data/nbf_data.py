"""Loaders for the visible gym data (training dialogues and benign train/dev prompts).

All embeddings are all-mpnet-base-v2 sentence embeddings (unit norm, 768-d)
stored as little-endian float16 and returned as float32 torch tensors.

Visible data:
  * data/downloads/nbf_cb/    multi-turn jailbreak dialogues against GPT-3.5-turbo
                              (Circuit Breakers goals; ActorAttack, Acronym,
                              Opposite-day, Crescendo) with GPT-4o 1-5 scores per
                              turn, query (u) and response (y) embeddings.
                              Splits: "train" and "dev".
  * data/benign/              benign / harmful prompts from MMLU, MT-Bench, XSTest
                              and JailbreakBench (query embeddings only).
                              Splits: "train" and "dev".
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

DIM = 768
ROOT = Path(__file__).resolve().parent
CB_DIR = ROOT / "downloads" / "nbf_cb"
BENIGN_DIR = ROOT / "benign"
UNSAFE_SCORE = 5
FAMILIES = ("actorattack", "acronym", "opposite_day", "crescendo")


def read_f16(path: Path, n_rows: int) -> torch.Tensor:
    data = bytearray(Path(path).read_bytes())
    t = torch.frombuffer(data, dtype=torch.float16)
    if t.numel() != n_rows * DIM:
        raise ValueError(f"{path}: expected {n_rows}x{DIM} values, found {t.numel()}")
    return t.view(n_rows, DIM).float()


def load_cb_dialogues(split: str, with_responses: bool = True) -> list[dict]:
    """Return dialogues of one split ("train" or "dev") as dicts with u, y, scores tensors."""
    meta = json.loads((CB_DIR / "dialogues.json").read_text(encoding="utf-8"))
    n = meta["n_turns"]
    u_all = read_f16(CB_DIR / "u.f16", n)
    y_all = read_f16(CB_DIR / "y.f16", n) if with_responses else None
    out = []
    for d in meta["dialogues"]:
        if d["split"] != split:
            continue
        sl = slice(d["offset"], d["offset"] + d["length"])
        out.append(
            {
                "id": d["id"],
                "family": d["family"],
                "u": u_all[sl],
                "y": y_all[sl] if y_all is not None else None,
                "scores": torch.tensor(d["scores"], dtype=torch.long),
            }
        )
    return out


def pad_dialogues(dialogues: list[dict], max_len: int | None = None):
    """Stack dialogues into padded tensors.

    Returns u [N,T,768], y [N,T,768], labels [N,T] (score - 1, 0 on padding), mask [N,T].
    """
    max_len = max_len or max(len(d["scores"]) for d in dialogues)
    n = len(dialogues)
    u = torch.zeros(n, max_len, DIM)
    y = torch.zeros(n, max_len, DIM)
    labels = torch.zeros(n, max_len, dtype=torch.long)
    mask = torch.zeros(n, max_len)
    for i, d in enumerate(dialogues):
        length = len(d["scores"])
        u[i, :length] = d["u"]
        if d.get("y") is not None:
            y[i, :length] = d["y"]
        labels[i, :length] = d["scores"] - 1
        mask[i, :length] = 1
    return u, y, labels, mask


def load_benign(split: str) -> list[dict]:
    """Return the benign-traffic sessions of one split ("train" or "dev").

    Each session: id, kind ("helpfulness" or "safety"), turns (texts), u [L,768] and
    tags: one dict per turn with "source" (mmlu / mtbench / xstest / jbb / context)
    and "label" ("benign" or "harmful"). Helpfulness sessions chain 1-8 MMLU /
    MT-Bench turns; safety sessions ask one XSTest or JailbreakBench prompt after
    0-7 benign MMLU context turns.
    """
    rows = [json.loads(line) for line in (BENIGN_DIR / f"benign_{split}.jsonl").read_text(encoding="utf-8").splitlines()]
    n = sum(r["length"] for r in rows)
    u_all = read_f16(BENIGN_DIR / f"benign_{split}_u.f16", n)
    return [
        {
            "id": r["id"],
            "kind": r["kind"],
            "tags": r["tags"],
            "turns": r["turns"],
            "u": u_all[r["offset"] : r["offset"] + r["length"]],
        }
        for r in rows
    ]

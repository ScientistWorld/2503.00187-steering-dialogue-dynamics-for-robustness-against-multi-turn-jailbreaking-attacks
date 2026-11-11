#!/usr/bin/env python3
"""Convert the authors' released Circuit-Breakers dialogue embeddings into plain arrays.

The NBF authors released the multi-turn jailbreak dialogues they trained on as
torch pickles inside ``train_val_embedding_data.zip`` (all-mpnet-base-v2 query
and response embeddings plus GPT-4o 1-5 safety scores per turn). This script
uses only the Python standard library and a *restricted* unpickler (no code
execution: only tensor-rebuild records are accepted) to write:

    <out>/u.f16          float16 [n_turns, 768] user-query embeddings
    <out>/y.f16          float16 [n_turns, 768] assistant-response embeddings
    <out>/dialogues.json per-dialogue family, split, offset, length, scores

Only the Circuit Breakers members of the archive are read. The archive's other
members are never extracted here.

Split rule (deterministic, visible to everyone): for Acronym / Opposite-day /
Crescendo the archive stores 1000 dialogues per family, one per Circuit
Breakers goal in goal order; goals 0-149 form the ``dev`` split. ActorAttack
stores several attempts per goal in goal order; its first 350 dialogues form
the ``dev`` split. Everything else is ``train``.
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import pickle
import struct
import zipfile
from pathlib import Path

DIM = 768
OTHERS_FAMILIES = ("acronym", "opposite_day", "crescendo")  # archive block order
OTHERS_BLOCK = 1000
OTHERS_DEV_GOALS = 150
ACTOR_DEV_DIALOGUES = 350


class _Storage:
    def __init__(self, key: str, numel: int):
        self.key, self.numel = key, numel


class _Tensor:
    def __init__(self, storage, offset, size, stride):
        self.storage, self.offset, self.size, self.stride = storage, offset, tuple(size), tuple(stride)


def _rebuild_tensor_v2(storage, offset, size, stride, *_args):
    return _Tensor(storage, offset, size, stride)


class _StorageType:
    def __init__(self, name: str):
        self.name = name


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild_tensor_v2
        if module == "torch" and name == "FloatStorage":
            return _StorageType(name)
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        raise pickle.UnpicklingError(f"refusing to load {module}.{name}")

    def persistent_load(self, pid):
        kind, stype, key, _location, numel = pid
        if kind != "storage" or getattr(stype, "name", "") != "FloatStorage":
            raise pickle.UnpicklingError(f"unexpected storage record {pid!r}")
        return _Storage(key, numel)


def read_release_member(archive: zipfile.ZipFile, member: str):
    """Return (trajectories, vector_reader) for one torch zip member of the archive."""
    inner = zipfile.ZipFile(io.BytesIO(archive.read(member)))
    pkl = [n for n in inner.namelist() if n.endswith("/data.pkl")][0]
    prefix = pkl[: -len("/data.pkl")]
    if inner.read(prefix + "/byteorder").strip() != b"little":
        raise SystemExit(f"{member}: unexpected byte order")
    with inner.open(pkl) as f:
        trajectories = _RestrictedUnpickler(f).load()

    def vector(t: _Tensor) -> tuple:
        if t.size != (DIM,) or t.stride != (1,):
            raise SystemExit(f"{member}: unexpected tensor layout {t.size} {t.stride}")
        raw = inner.read(f"{prefix}/data/{t.storage.key}")
        start = 4 * t.offset
        return struct.unpack(f"<{DIM}f", raw[start : start + 4 * DIM])

    return trajectories, vector


def convert(zip_path: Path, out_dir: Path) -> dict:
    archive = zipfile.ZipFile(zip_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    pack = struct.Struct(f"<{DIM}e").pack
    dialogues = []
    offset = 0
    with (out_dir / "u.f16").open("wb") as fu, (out_dir / "y.f16").open("wb") as fy:
        for member in ("circuit_breakers_others.pt", "circuit_breakers_actorattack.pt"):
            trajectories, vector = read_release_member(archive, member)
            for index, traj in enumerate(trajectories):
                if member.endswith("others.pt"):
                    family = OTHERS_FAMILIES[index // OTHERS_BLOCK]
                    goal = index % OTHERS_BLOCK
                    split = "dev" if goal < OTHERS_DEV_GOALS else "train"
                else:
                    family = "actorattack"
                    goal = None
                    split = "dev" if index < ACTOR_DEV_DIALOGUES else "train"
                scores = []
                for step in traj:
                    fu.write(pack(*vector(step["u"])))
                    fy.write(pack(*vector(step["y"])))
                    score = int(step["score"])
                    if not 1 <= score <= 5:
                        raise SystemExit(f"{member}[{index}]: score {score} outside 1-5")
                    scores.append(score)
                dialogues.append(
                    {
                        "id": f"cb_{family}_{index % OTHERS_BLOCK if goal is not None else index:04d}",
                        "family": family,
                        "goal": goal,
                        "split": split,
                        "offset": offset,
                        "length": len(scores),
                        "scores": scores,
                    }
                )
                offset += len(scores)
    meta = {
        "source": "NBF-LLM authors' train_val_embedding_data.zip (Circuit Breakers members only)",
        "embedding_model": "sentence-transformers/all-mpnet-base-v2 (unit-norm, 768-d), stored as float16",
        "score_scale": "GPT-4o judge score 1-5 per turn; 5 = jailbroken (unsafe) response, 1-4 = safe set",
        "dim": DIM,
        "n_turns": offset,
        "split_rule": __doc__.split("Split rule")[1].strip(),
        "dialogues": dialogues,
    }
    (out_dir / "dialogues.json").write_text(json.dumps(meta) + "\n", encoding="utf-8")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", default="data/downloads/nbf_release_zip/train_val_embedding_data.zip")
    parser.add_argument("--out-dir", default="data/downloads/nbf_cb")
    args = parser.parse_args()
    meta = convert(Path(args.zip), Path(args.out_dir))
    counts = collections.Counter((d["family"], d["split"]) for d in meta["dialogues"])
    print(json.dumps({f"{k[0]}/{k[1]}": v for k, v in sorted(counts.items())}, indent=1))
    print("turns:", meta["n_turns"])


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build the hidden multi-turn attack test set from the authors' release archive.

Reads only the HarmBench members of ``train_val_embedding_data.zip`` (the
authors' GPT-3.5-turbo multi-turn attack dialogues on the 200 HarmBench goals,
with GPT-4o 1-5 judge scores per turn) through a restricted unpickler that
accepts tensor records only, and writes:

    eval/test/data/hb_u.f16           float16 [n_turns, 768] user-query embeddings
    eval/test/data/hb_dialogues.json  family, offset, length, per-turn scores

Response embeddings are not kept: guards never see them.
The archive stores Acronym, Opposite-day and Crescendo as consecutive blocks
of 200 dialogues (block ASRs 0.905 / 0.785 / 0.560; the last two equal the
paper's Table 1 GPT-3.5 Opposite-day and Crescendo ASRs).
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
OTHERS_FAMILIES = ("acronym", "opposite_day", "crescendo")
OTHERS_BLOCK = 200


class _Storage:
    def __init__(self, key, numel):
        self.key, self.numel = key, numel


class _Tensor:
    def __init__(self, storage, offset, size, stride):
        self.storage, self.offset, self.size, self.stride = storage, offset, tuple(size), tuple(stride)


def _rebuild(storage, offset, size, stride, *_args):
    return _Tensor(storage, offset, size, stride)


class _Kind:
    def __init__(self, name):
        self.name = name


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return _rebuild
        if (module, name) == ("torch", "FloatStorage"):
            return _Kind(name)
        if (module, name) == ("collections", "OrderedDict"):
            return collections.OrderedDict
        raise pickle.UnpicklingError(f"refusing {module}.{name}")

    def persistent_load(self, pid):
        kind, stype, key, _loc, numel = pid
        if kind != "storage" or getattr(stype, "name", "") != "FloatStorage":
            raise pickle.UnpicklingError(f"unexpected storage {pid!r}")
        return _Storage(key, numel)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True)
    parser.add_argument("--out-dir", default="eval/test/data")
    args = parser.parse_args()
    archive = zipfile.ZipFile(args.zip)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pack = struct.Struct(f"<{DIM}e").pack
    dialogues, offset = [], 0
    with (out / "hb_u.f16").open("wb") as fu:
        for member in ("harmbench_others.pt", "harmbench_actorattack.pt"):
            inner = zipfile.ZipFile(io.BytesIO(archive.read(member)))
            pkl = [n for n in inner.namelist() if n.endswith("/data.pkl")][0]
            prefix = pkl[: -len("/data.pkl")]
            with inner.open(pkl) as f:
                trajectories = _Unpickler(f).load()
            for index, traj in enumerate(trajectories):
                if member.endswith("others.pt"):
                    family, local = OTHERS_FAMILIES[index // OTHERS_BLOCK], index % OTHERS_BLOCK
                else:
                    family, local = "actorattack", index
                scores = []
                for step in traj:
                    t = step["u"]
                    if t.size != (DIM,) or t.stride != (1,):
                        raise SystemExit("unexpected tensor layout")
                    raw = inner.read(f"{prefix}/data/{t.storage.key}")
                    fu.write(pack(*struct.unpack(f"<{DIM}f", raw[4 * t.offset : 4 * (t.offset + DIM)])))
                    scores.append(int(step["score"]))
                dialogues.append(
                    {"id": f"hb_{family}_{local:04d}", "family": family, "offset": offset,
                     "length": len(scores), "scores": scores}
                )
                offset += len(scores)
    meta = {
        "source": "NBF-LLM authors' train_val_embedding_data.zip (HarmBench members)",
        "embedding_model": "sentence-transformers/all-mpnet-base-v2, float16",
        "dim": DIM,
        "n_turns": offset,
        "dialogues": dialogues,
    }
    (out / "hb_dialogues.json").write_text(json.dumps(meta) + "\n", encoding="utf-8")
    fam = collections.Counter(d["family"] for d in dialogues)
    asr = {f: sum(5 in d["scores"] for d in dialogues if d["family"] == f) / n for f, n in fam.items()}
    print(json.dumps({"dialogues": dict(fam), "unguarded_asr": asr, "turns": offset}, indent=1))


if __name__ == "__main__":
    main()

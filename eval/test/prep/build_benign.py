#!/usr/bin/env python3
"""Split the paper's helpfulness / over-refusal benchmarks into train, dev and hidden test.

Sources (all public; the paper evaluates helpfulness on MMLU and MT-Bench and
over-refusal on XSTest and JailbreakBench benign behaviors):

  * MMLU test questions (hendrycks/test CSVs), rendered as question + 4 choices
  * MT-Bench 80 two-turn questions (lm-sys/FastChat question.jsonl)
  * XSTest prompts (250 safe, 200 unsafe contrast prompts)
  * JailbreakBench JBB-Behaviors (100 benign, 100 harmful goals)

Every item gets a deterministic SHA-256 rank inside its (source, label) group;
the lowest ranks go to the hidden test split, the next to dev, the next to
train. Train and dev are written to ``data/benign/``; test stays in
``eval/test/data/``.

Usage (maintainer, needs internet):
    python eval/test/prep/build_benign.py --raw-dir tmp/benign_raw --mmlu-dir <hendrycks data/test>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

SOURCES = {
    "xstest_prompts.csv": "https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv",
    "mt_bench_question.jsonl": "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl",
    "jbb_benign.csv": "https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors/resolve/main/data/benign-behaviors.csv",
    "jbb_harmful.csv": "https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors/resolve/main/data/harmful-behaviors.csv",
}
MMLU_TAR = "https://people.eecs.berkeley.edu/~hendrycks/data.tar"

# (source, label) -> (test, dev, train) counts
ALLOCATION = {
    ("mmlu", "benign"): (600, 300, 300),
    ("mtbench", "benign"): (50, 15, 15),
    ("xstest", "benign"): (150, 50, 50),
    ("xstest", "harmful"): (120, 40, 40),
    ("jbb", "benign"): (60, 20, 20),
    ("jbb", "harmful"): (60, 20, 20),
}


def fetch(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        if not (raw_dir / name).exists():
            subprocess.run(["curl", "-sfL", url, "-o", str(raw_dir / name)], check=True)


def rank_key(item_id: str) -> str:
    return hashlib.sha256(f"nbf-gym-benign-split/{item_id}".encode()).hexdigest()


def load_items(raw_dir: Path, mmlu_dir: Path) -> list[dict]:
    items = []
    for path in sorted(mmlu_dir.glob("*_test.csv")):
        subject = path.name[: -len("_test.csv")]
        with path.open(encoding="utf-8") as f:
            for idx, row in enumerate(csv.reader(f)):
                q, a, b, c, d = row[:5]
                text = f"{q}\nA. {a}\nB. {b}\nC. {c}\nD. {d}"
                items.append({"id": f"mmlu_{subject}_{idx:04d}", "source": "mmlu", "label": "benign", "turns": [text]})
    with (raw_dir / "mt_bench_question.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            items.append(
                {"id": f"mtbench_{row['question_id']}", "source": "mtbench", "label": "benign",
                 "category": row["category"], "turns": list(row["turns"])}
            )
    with (raw_dir / "xstest_prompts.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            items.append(
                {"id": f"xstest_{int(row['id']):03d}", "source": "xstest",
                 "label": "benign" if row["label"] == "safe" else "harmful",
                 "category": row["type"], "turns": [row["prompt"]]}
            )
    for name, label in (("jbb_benign.csv", "benign"), ("jbb_harmful.csv", "harmful")):
        with (raw_dir / name).open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                items.append(
                    {"id": f"jbb_{label}_{int(row['Index']):03d}", "source": "jbb", "label": label,
                     "category": row["Category"], "turns": [row["Goal"]]}
                )
    return items


def split(items: list[dict]) -> dict[str, list[dict]]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for item in items:
        groups.setdefault((item["source"], item["label"]), []).append(item)
    out = {"test": [], "dev": [], "train": []}
    for key, (n_test, n_dev, n_train) in ALLOCATION.items():
        ranked = sorted(groups[key], key=lambda it: rank_key(it["id"]))
        if len(ranked) < n_test + n_dev + n_train:
            raise SystemExit(f"{key}: only {len(ranked)} items")
        out["test"] += ranked[:n_test]
        out["dev"] += ranked[n_test : n_test + n_dev]
        out["train"] += ranked[n_test + n_dev : n_test + n_dev + n_train]
    for name in out:
        out[name].sort(key=lambda it: it["id"])
    return out


def compose_sessions(items: list[dict], split_name: str) -> list[dict]:
    """Turn one split's prompts into multi-turn sessions so that turn position carries no label.

    * helpfulness sessions: MMLU questions and MT-Bench two-turn questions, merged in
      hash order and packed into sessions of 1..8 turns (target length 1 + j mod 8);
      every turn is scored for helpfulness.
    * safety sessions: each XSTest / JailbreakBench prompt is asked after k = hash mod 8
      benign MMLU context turns of the same split (context turns are not scored).
    """
    pool = sorted([it for it in items if it["source"] in ("mmlu", "mtbench")], key=lambda it: rank_key(it["id"]))
    mmlu = [it for it in pool if it["source"] == "mmlu"]
    sessions = []
    j, cur_turns, cur_tags = 0, [], []
    for it in pool:
        for text in it["turns"]:
            cur_turns.append(text)
            cur_tags.append({"source": it["source"], "label": "benign", "item": it["id"]})
        if len(cur_turns) >= 1 + (j % 8):
            sessions.append({"id": f"{split_name}_help_{j:04d}", "kind": "helpfulness", "turns": cur_turns, "tags": cur_tags})
            j, cur_turns, cur_tags = j + 1, [], []
    if cur_turns:
        sessions.append({"id": f"{split_name}_help_{j:04d}", "kind": "helpfulness", "turns": cur_turns, "tags": cur_tags})
    probes = sorted([it for it in items if it["source"] in ("xstest", "jbb")], key=lambda it: rank_key(it["id"]))
    ptr = 0
    for it in probes:
        k = int(rank_key("prefix/" + it["id"])[:8], 16) % 8
        prefix = [mmlu[(ptr + i) % len(mmlu)] for i in range(k)]
        ptr += k
        sessions.append(
            {
                "id": f"{split_name}_{it['id']}",
                "kind": "safety",
                "turns": [p["turns"][0] for p in prefix] + [it["turns"][0]],
                "tags": [{"source": "context", "label": "benign", "item": p["id"]} for p in prefix]
                + [{"source": it["source"], "label": it["label"], "item": it["id"], "category": it.get("category")}],
            }
        )
    return sessions


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    offset = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            row = dict(row, offset=offset, length=len(row["turns"]))
            offset += len(row["turns"])
            f.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="tmp/benign_raw")
    parser.add_argument("--mmlu-dir", default=None, help="directory of MMLU *_test.csv (downloaded if omitted)")
    parser.add_argument("--train-out", default="data/benign/benign_train.jsonl")
    parser.add_argument("--dev-out", default="data/benign/benign_dev.jsonl")
    parser.add_argument("--test-out", default="eval/test/data/benign_test.jsonl")
    args = parser.parse_args()
    raw = Path(args.raw_dir)
    fetch(raw)
    mmlu_dir = Path(args.mmlu_dir) if args.mmlu_dir else raw / "mmlu" / "data" / "test"
    if not mmlu_dir.exists():
        (raw / "mmlu").mkdir(parents=True, exist_ok=True)
        subprocess.run(["curl", "-sfL", MMLU_TAR, "-o", str(raw / "mmlu" / "data.tar")], check=True)
        subprocess.run(["tar", "-xf", "data.tar"], cwd=raw / "mmlu", check=True)
    parts = split(load_items(raw, mmlu_dir))
    outs = {"train": args.train_out, "dev": args.dev_out, "test": args.test_out}
    for name, rows in parts.items():
        sessions = compose_sessions(rows, name)
        write_jsonl(Path(outs[name]), sessions)
        kinds = {k: sum(s["kind"] == k for s in sessions) for k in ("helpfulness", "safety")}
        print(name, len(rows), "prompts ->", len(sessions), "sessions", kinds,
              sum(len(s["turns"]) for s in sessions), "turns")


if __name__ == "__main__":
    main()

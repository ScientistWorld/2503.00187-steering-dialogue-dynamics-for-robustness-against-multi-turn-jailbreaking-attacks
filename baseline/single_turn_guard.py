"""Single-turn guard baseline: classify each user query on its own, ignoring dialogue context.

A lightweight prompt classifier (768 -> 32 -> 32 -> 5 ReLU MLP over the query
embedding) trained with plain cross-entropy on every turn of the training
dialogues to predict the 1-5 judge score. A query is filtered iff
p(score 5) - max_{s<5} p(s) + eta >= 0. Its state never changes, so it sees
each turn in isolation, like the single-turn moderation filters the paper
compares against.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.nbf_data import DIM, load_cb_dialogues  # noqa: E402

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
torch.use_deterministic_algorithms(True, warn_only=True)


class SingleTurnClassifier(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 5),
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.mlp(u)


class SingleTurnGuard(nn.Module):
    def __init__(self, clf: SingleTurnClassifier, eta: float):
        super().__init__()
        self.clf = clf
        self.register_buffer("eta", torch.tensor(float(eta)))
        self.register_buffer("initial_state", torch.zeros(1))

    def forward(self, state: torch.Tensor, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        probs = torch.softmax(self.clf(u), dim=-1)
        return probs[:, -1] - probs[:, :-1].max(dim=-1).values + self.eta, state


def flat_turns(split: str):
    dialogues = load_cb_dialogues(split, with_responses=False)
    return torch.cat([d["u"] for d in dialogues]), torch.cat([d["scores"] for d in dialogues]) - 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="tmp/outputs/guards/single_turn_guard.pt")
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eta", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sweep-etas", type=float, nargs="*", default=[],
                        help="also export the same classifier at these thresholds (trade-off curve)")
    parser.add_argument("--sweep-prefix", default=None, help="path prefix for sweep exports: <prefix>_eta<eta>.pt")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    u, y = (t.to(device) for t in flat_turns("train"))
    vu, vy = (t.to(device) for t in flat_turns("dev"))
    clf = SingleTurnClassifier(args.hidden).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=args.lr)
    best, best_state = float("inf"), None
    for epoch in range(args.epochs):
        clf.train()
        for idx in torch.randperm(u.shape[0], generator=gen).split(args.batch_size):
            loss = nn.functional.cross_entropy(clf(u[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        clf.eval()
        with torch.no_grad():
            vl = nn.functional.cross_entropy(clf(vu), vy).item()
        if vl < best:
            best = vl
            best_state = {k: v.detach().cpu().clone() for k, v in clf.state_dict().items()}
        if (epoch + 1) % 20 == 0:
            print(f"single-turn epoch {epoch + 1} val_ce {vl:.4f} best {best:.4f}", flush=True)
    clf = SingleTurnClassifier(args.hidden)
    clf.load_state_dict(best_state)
    guard = torch.jit.script(SingleTurnGuard(clf.eval(), args.eta).eval())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    guard.save(args.out)
    print(f"exported {args.out}")
    for eta in args.sweep_etas:
        tag = f"{eta:g}".replace("-", "m").replace(".", "p")
        path = f"{args.sweep_prefix}_eta{tag}.pt"
        torch.jit.script(SingleTurnGuard(clf.eval(), eta).eval()).save(path)
        print(f"exported {path}")


if __name__ == "__main__":
    main()

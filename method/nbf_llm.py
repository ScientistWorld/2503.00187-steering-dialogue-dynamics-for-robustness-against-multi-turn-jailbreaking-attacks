"""Neural dialogue dynamics + neural barrier function (NBF) safety steering.

Reproduction of Hu, Robey & Liu, "Steering Dialogue Dynamics for Robustness
against Multi-turn Jailbreaking Attacks" (arXiv 2503.00187), following the
authors' released train.py / build_ssm_nbf.py:

  stage 1  neural state-space model (Eq. 3-4)
           x_k = f(x_{k-1}, u_k),  z_k = g(x_k, u_k),  x_0 = 0
           f, g: 3-layer ReLU MLPs (2*768 -> 512 -> 512 -> 768), MSE on response
           embeddings, Adam 1e-4, 200 epochs, best validation-MSE checkpoint.
  stage 2  safety predictor h(x_{k-1}, u_k) (2*768 -> 32 -> 32 -> 5 classes)
           with frozen dynamics, loss = CE + 100 * L_SI + 100 * L_SS
           (Eqs. 6, 11, 12 as implemented by the authors; kappa = 3,
           training eta = 0), Adam 1e-3, 200 epochs, best validation-CE checkpoint.
  steering barrier margin h = p(unsafe) - max_safe p(safe class); a query is
           filtered iff h(x_{k-1}, u_k) + eta >= 0, and a filtered query never
           enters the dialogue state (state is not advanced).

The exported guard is a TorchScript module following the gym's guard interface
(see briefing/evaluation.md): buffer ``initial_state`` and
``forward(state, u) -> (score, next_state)``; filtered iff score >= 0.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.nbf_data import DIM, load_benign, load_cb_dialogues, pad_dialogues  # noqa: E402

NUM_CLASSES = 5
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
torch.use_deterministic_algorithms(True, warn_only=True)


class NeuralStateSpaceModel(nn.Module):
    def __init__(self, state_dim: int, input_dim: int, output_dim: int, hidden_dim: int):
        super().__init__()
        self.state_transition = nn.Sequential(
            nn.Linear(state_dim + input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )
        self.observation_model = nn.Sequential(
            nn.Linear(state_dim + input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x_prev: torch.Tensor, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.state_transition(torch.cat([x_prev, u], dim=-1))
        z = self.observation_model(torch.cat([x, u], dim=-1))
        return x, z


class NeuralBarrierFunction(nn.Module):
    def __init__(self, state_dim: int, input_dim: int, hidden_dim: int, class_num: int = NUM_CLASSES):
        super().__init__()
        self.nbf = nn.Sequential(
            nn.Linear(state_dim + input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, class_num),
        )

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.nbf(torch.cat([x, u], dim=-1))


def barrier_margin(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    return probs[..., -1] - probs[..., :-1].max(dim=-1).values


def loss_forward_invariance(logits: torch.Tensor, m: torch.Tensor, label: torch.Tensor | None = None) -> torch.Tensor:
    """Authors' loss_forward_invariance: L_SI when label is None, L_SS otherwise."""
    margin = barrier_margin(logits)
    if label is None:
        loss = torch.relu(margin) * m
    else:
        safe = (label != NUM_CLASSES - 1).float()
        loss = torch.relu((2 * safe - 1) * margin) * m
    if m.sum() == 0:
        return torch.zeros((), device=logits.device)
    return loss.sum() / m.sum()


# --------------------------------------------------------------------------- data

def tensors(split: str, device, extra: list[dict] | None = None):
    dialogues = load_cb_dialogues(split)
    if extra:
        dialogues = dialogues + extra
    u, y, labels, mask = pad_dialogues(dialogues)
    return u.to(device), y.to(device), labels.to(device), mask.to(device)


def benign_training_dialogues(benign_score: int, harmful_score: int) -> list[dict]:
    """Optional extra training dialogues from the visible benign *train* prompts (no responses)."""
    out = []
    for item in load_benign("train"):
        scores = [harmful_score if tag["label"] == "harmful" else benign_score for tag in item["tags"]]
        out.append({"id": item["id"], "family": "benign", "u": item["u"], "y": None,
                    "scores": torch.tensor(scores, dtype=torch.long)})
    return out


def batches(n: int, batch_size: int, generator: torch.Generator):
    perm = torch.randperm(n, generator=generator)
    for i in range(0, n, batch_size):
        yield perm[i : i + batch_size]


# --------------------------------------------------------------------------- stage 1

def ssm_rollout(ssm, u):
    x = torch.zeros(u.shape[0], u.shape[-1], device=u.device)
    preds = []
    for t in range(u.shape[1]):
        x, z = ssm(x, u[:, t])
        preds.append(z)
    return torch.stack(preds, 1)


def train_ssm(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    u, y, _, mask = tensors("train", device)
    vu, vy, _, vmask = tensors("dev", device)
    ssm = NeuralStateSpaceModel(DIM, DIM, DIM, args.hidden).to(device)
    opt = torch.optim.Adam(ssm.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    m3, vm3 = mask.unsqueeze(-1), vmask.unsqueeze(-1)
    for epoch in range(args.epochs):
        ssm.train()
        total, nb = 0.0, 0
        t0 = time.time()
        for idx in batches(u.shape[0], args.batch_size, gen):
            pred = ssm_rollout(ssm, u[idx])
            loss = mse(pred * m3[idx], y[idx] * m3[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total, nb = total + loss.item(), nb + 1
        ssm.eval()
        with torch.no_grad():
            vl, vn = 0.0, 0
            for i in range(0, vu.shape[0], args.batch_size):
                sl = slice(i, i + args.batch_size)
                vl += mse(ssm_rollout(ssm, vu[sl]) * vm3[sl], vy[sl] * vm3[sl]).item()
                vn += 1
            vl /= max(1, vn)
        if vl < best:
            best = vl
            torch.save({"ssm": ssm.state_dict(), "hidden": args.hidden, "epoch": epoch + 1, "val_mse": vl}, out)
        print(f"ssm epoch {epoch + 1}/{args.epochs} train {total / max(1, nb):.6f} val {vl:.6f} best {best:.6f} ({time.time() - t0:.1f}s)", flush=True)


# --------------------------------------------------------------------------- stage 2

def nbf_forward(ssm, nbf, u):
    """Return (logits [B,T,5], next-state logits [B,T,5]) exactly as the authors' loop."""
    b, steps, _ = u.shape
    x = torch.zeros(b, u.shape[-1], device=u.device)
    preds, preds_next = [], []
    for t in range(steps):
        x_prev = x
        with torch.no_grad():  # dynamics are frozen in stage 2
            x, _ = ssm(x_prev, u[:, t])
        u_next = u[:, t + 1] if t < steps - 1 else u[:, t]
        preds.append(nbf(x_prev, u[:, t]))
        preds_next.append(nbf(x, u_next))
    return torch.stack(preds, 1), torch.stack(preds_next, 1)


def nbf_losses(ssm, nbf, u, labels, mask, kappa):
    logits, logits_next = nbf_forward(ssm, nbf, u)
    fi_mask = torch.zeros_like(mask)
    fi_mask[:, :-kappa] = mask[:, kappa:]
    m = mask.unsqueeze(-1)
    # Authors' masked cross entropy: padded positions contribute zero logits and label 0.
    ce = nn.functional.cross_entropy((logits * m).reshape(-1, NUM_CLASSES), (labels * mask.long()).reshape(-1))
    si = loss_forward_invariance(logits_next, fi_mask)
    ss = loss_forward_invariance(logits, mask, labels)
    return ce, si, ss


def train_nbf(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    extra = benign_training_dialogues(args.benign_score, args.harmful_score) if args.benign_train else None
    u, _, labels, mask = tensors("train", device, extra)
    vu, _, vlabels, vmask = tensors("dev", device)
    ssm_ckpt = torch.load(args.ssm, map_location=device)
    ssm = NeuralStateSpaceModel(DIM, DIM, DIM, ssm_ckpt["hidden"]).to(device)
    ssm.load_state_dict(ssm_ckpt["ssm"])
    ssm.eval()
    for p in ssm.parameters():
        p.requires_grad_(False)
    nbf = NeuralBarrierFunction(DIM, DIM, args.hidden).to(device)
    opt = torch.optim.Adam(nbf.parameters(), lr=args.lr)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    for epoch in range(args.epochs):
        nbf.train()
        sums = torch.zeros(4)
        nb = 0
        for idx in batches(u.shape[0], args.batch_size, gen):
            ce, si, ss = nbf_losses(ssm, nbf, u[idx], labels[idx], mask[idx], args.kappa)
            loss = ce + args.si_weight * si + args.ss_weight * ss
            opt.zero_grad()
            loss.backward()
            opt.step()
            sums += torch.tensor([loss.item(), ce.item(), si.item(), ss.item()])
            nb += 1
        nbf.eval()
        with torch.no_grad():
            vce, vn = 0.0, 0
            for i in range(0, vu.shape[0], args.batch_size):
                sl = slice(i, i + args.batch_size)
                ce, _, _ = nbf_losses(ssm, nbf, vu[sl], vlabels[sl], vmask[sl], args.kappa)
                vce += ce.item()
                vn += 1
            vce /= max(1, vn)
        if vce < best:
            best = vce
            torch.save(
                {"ssm": ssm.state_dict(), "nbf": nbf.state_dict(), "ssm_hidden": ssm_ckpt["hidden"],
                 "nbf_hidden": args.hidden, "epoch": epoch + 1, "val_ce": vce,
                 "config": {k: v for k, v in vars(args).items() if k != "func"}},
                out,
            )
        s = sums / max(1, nb)
        print(f"nbf epoch {epoch + 1}/{args.epochs} loss {s[0]:.4f} ce {s[1]:.4f} si {s[2]:.4f} ss {s[3]:.4f} val_ce {vce:.4f} best {best:.4f}", flush=True)


# --------------------------------------------------------------------------- export

class NBFGuard(nn.Module):
    """TorchScript guard: score = h(x_{k-1}, u_k) + eta, next_state = f(x_{k-1}, u_k)."""

    def __init__(self, ssm: NeuralStateSpaceModel, nbf: NeuralBarrierFunction, eta: float):
        super().__init__()
        self.ssm = ssm
        self.nbf = nbf
        self.register_buffer("eta", torch.tensor(float(eta)))
        self.register_buffer("initial_state", torch.zeros(DIM))

    def forward(self, state: torch.Tensor, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        probs = torch.softmax(self.nbf(state, u), dim=-1)
        margin = probs[:, -1] - probs[:, :-1].max(dim=-1).values
        next_state, _ = self.ssm(state, u)
        return margin + self.eta, next_state


class _Pair(nn.Module):
    def __init__(self, ssm: NeuralStateSpaceModel, nbf: NeuralBarrierFunction):
        super().__init__()
        self.ssm = ssm
        self.nbf = nbf

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        probs = torch.softmax(self.nbf(x, u), dim=-1)
        margin = probs[:, -1] - probs[:, :-1].max(dim=-1).values
        nx, _ = self.ssm(x, u)
        return margin, nx


class NBFEnsembleGuard(nn.Module):
    """Average barrier margin over independently trained (dynamics, barrier) pairs."""

    def __init__(self, ssms: list, nbfs: list, eta: float):
        super().__init__()
        self.pairs = nn.ModuleList([_Pair(s, n) for s, n in zip(ssms, nbfs)])
        self.k = len(ssms)
        self.register_buffer("eta", torch.tensor(float(eta)))
        self.register_buffer("initial_state", torch.zeros(DIM * len(ssms)))

    def forward(self, state: torch.Tensor, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        parts = torch.chunk(state, self.k, dim=-1)
        margins = []
        nexts = []
        for i, pair in enumerate(self.pairs):
            m, nx = pair(parts[i], u)
            margins.append(m)
            nexts.append(nx)
        return torch.stack(margins, 0).mean(0) + self.eta, torch.cat(nexts, dim=-1)


def load_pair(path):
    ckpt = torch.load(path, map_location="cpu")
    ssm = NeuralStateSpaceModel(DIM, DIM, DIM, ckpt["ssm_hidden"])
    ssm.load_state_dict(ckpt["ssm"])
    nbf = NeuralBarrierFunction(DIM, DIM, ckpt["nbf_hidden"])
    nbf.load_state_dict(ckpt["nbf"])
    return ssm.eval(), nbf.eval()


def export(args) -> None:
    pairs = [load_pair(p) for p in args.model]
    if len(pairs) == 1:
        guard = NBFGuard(pairs[0][0], pairs[0][1], args.eta)
    else:
        guard = NBFEnsembleGuard([p[0] for p in pairs], [p[1] for p in pairs], args.eta)
    scripted = torch.jit.script(guard.eval())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    scripted.save(args.out)
    n = sum(p.numel() for p in guard.parameters())
    print(f"exported {args.out} (eta={args.eta}, {n / 1e6:.3f}M parameters)")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train-ssm")
    p.add_argument("--out", default="checkpoints/nbf/ssm_seed0.pt")
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=train_ssm)
    p = sub.add_parser("train-nbf")
    p.add_argument("--ssm", default="checkpoints/nbf/ssm_seed0.pt")
    p.add_argument("--out", default="checkpoints/nbf/nbf_seed0.pt")
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--kappa", type=int, default=3)
    p.add_argument("--si-weight", type=float, default=100.0)
    p.add_argument("--ss-weight", type=float, default=100.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--benign-train", action="store_true",
                   help="also train on data/benign train prompts (variant; not the paper's setting)")
    p.add_argument("--benign-score", type=int, default=1)
    p.add_argument("--harmful-score", type=int, default=5)
    p.set_defaults(func=train_nbf)
    p = sub.add_parser("export")
    p.add_argument("--model", nargs="+", required=True)
    p.add_argument("--eta", type=float, default=1e-3)
    p.add_argument("--out", required=True)
    p.set_defaults(func=export)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

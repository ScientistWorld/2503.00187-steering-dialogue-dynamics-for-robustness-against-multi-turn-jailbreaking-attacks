import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, Dataset, random_split


SAFE_CLASS_COUNT = 4
UNSAFE_CLASS = 4


class TrajectoryDataset(Dataset):
    def __init__(self, paths, embedder, cache_path=None):
        self.rows = []
        for path in paths:
            with Path(path).open(encoding="utf-8") as f:
                self.rows.extend(json.loads(line) for line in f)
        self.max_len = max(len(r["turns"]) for r in self.rows)
        self.embedder = embedder
        self.cache_path = Path(cache_path) if cache_path else None
        if self.cache_path and self.cache_path.exists():
            payload = torch.load(self.cache_path, map_location="cpu")
            self.encoded = payload["encoded"]
            return
        self.encoded = [self._encode_row(row) for row in self.rows]
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"encoded": self.encoded}, self.cache_path)

    def _encode_row(self, row):
        users = [t["user"] for t in row["turns"]]
        assistants = [t["assistant"] for t in row["turns"]]
        u = torch.tensor(self.embedder.encode(users, normalize_embeddings=True), dtype=torch.float32)
        z = torch.tensor(self.embedder.encode(assistants, normalize_embeddings=True), dtype=torch.float32)
        labels = torch.tensor([int(t["score"]) - 1 for t in row["turns"]], dtype=torch.long)
        attacks = [t.get("attack", row.get("attack", "unknown")) for t in row["turns"]]
        return {"id": row["id"], "split": row["split"], "attack": row["attack"], "u": u, "z": z, "labels": labels, "attacks": attacks}

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        row = self.encoded[idx]
        dim = row["u"].shape[-1]
        length = row["u"].shape[0]
        u = torch.zeros(self.max_len, dim)
        z = torch.zeros(self.max_len, dim)
        labels = torch.zeros(self.max_len, dtype=torch.long)
        mask = torch.zeros(self.max_len)
        u[:length] = row["u"]
        z[:length] = row["z"]
        labels[:length] = row["labels"]
        mask[:length] = 1
        return u, z, labels, mask, idx


class NeuralStateSpaceModel(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.state_transition = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )
        self.observation_model = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x_prev, u):
        x = self.state_transition(torch.cat([x_prev, u], dim=-1))
        z_hat = self.observation_model(torch.cat([x, u], dim=-1))
        return x, z_hat


class NeuralBarrierFunction(nn.Module):
    def __init__(self, dim, hidden_dim, classes=5):
        super().__init__()
        self.nbf = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, classes),
        )

    def forward(self, x_prev, u):
        return self.nbf(torch.cat([x_prev, u], dim=-1))


def barrier_margin(logits):
    probs = torch.softmax(logits, dim=-1)
    return probs[..., UNSAFE_CLASS] - probs[..., :SAFE_CLASS_COUNT].max(dim=-1).values


def safe_set_loss(logits, labels, mask, eta):
    margin = barrier_margin(logits) + eta
    safe = (labels != UNSAFE_CLASS).float()
    sign = 2 * safe - 1
    return (torch.relu(sign * margin) * mask).sum() / mask.sum().clamp_min(1)


def invariance_loss(next_logits, mask, eta, kappa):
    margin = barrier_margin(next_logits) + eta
    inv_mask = torch.zeros_like(mask)
    if mask.shape[1] > kappa:
        inv_mask[:, :-kappa] = mask[:, kappa:]
    return (torch.relu(margin) * inv_mask).sum() / inv_mask.sum().clamp_min(1)


def rollout(ssm, nbf, u_batch):
    batch, steps, dim = u_batch.shape
    x = torch.zeros(batch, dim, device=u_batch.device)
    z_hats, logits, next_logits = [], [], []
    for t in range(steps):
        u = u_batch[:, t]
        x_prev = x
        logits.append(nbf(x_prev, u))
        x, z_hat = ssm(x_prev, u)
        z_hats.append(z_hat)
        u_next = u_batch[:, min(t + 1, steps - 1)]
        next_logits.append(nbf(x, u_next))
    return torch.stack(z_hats, 1), torch.stack(logits, 1), torch.stack(next_logits, 1)


def train(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedder = SentenceTransformer(args.embedding_model, device="cpu")
    dataset = TrajectoryDataset([args.train_data], embedder, args.cache)
    dim = dataset.encoded[0]["u"].shape[-1]
    train_len = max(1, int(len(dataset) * 0.9))
    val_len = len(dataset) - train_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    ssm = NeuralStateSpaceModel(dim, args.ssm_hidden).to(device)
    nbf = NeuralBarrierFunction(dim, args.nbf_hidden).to(device)
    opt_ssm = torch.optim.Adam(ssm.parameters(), lr=args.ssm_lr)
    mse = nn.MSELoss(reduction="none")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.ssm_epochs):
        ssm.train()
        total = 0.0
        for u, z, labels, mask, _ in train_loader:
            u, z, mask = u.to(device), z.to(device), mask.to(device)
            z_hat, _, _ = rollout(ssm, nbf, u)
            loss = (mse(z_hat, z).mean(-1) * mask).sum() / mask.sum().clamp_min(1)
            opt_ssm.zero_grad()
            loss.backward()
            opt_ssm.step()
            total += loss.item()
        print(f"ssm_epoch={epoch+1} loss={total / max(1, len(train_loader)):.6f}", flush=True)

    opt_nbf = torch.optim.Adam(nbf.parameters(), lr=args.nbf_lr)
    ce = nn.CrossEntropyLoss(reduction="none")
    for epoch in range(args.nbf_epochs):
        ssm.eval()
        nbf.train()
        total = 0.0
        for u, z, labels, mask, _ in train_loader:
            u, labels, mask = u.to(device), labels.to(device), mask.to(device)
            with torch.no_grad():
                _, _, _ = rollout(ssm, nbf, u)
            _, logits, next_logits = rollout(ssm, nbf, u)
            ce_loss = (ce(logits.reshape(-1, 5), labels.reshape(-1)).reshape_as(mask) * mask).sum() / mask.sum().clamp_min(1)
            ss_loss = safe_set_loss(logits, labels, mask, args.train_eta)
            si_loss = invariance_loss(next_logits, mask, args.train_eta, args.kappa)
            loss = ce_loss + args.safe_weight * ss_loss + args.invariance_weight * si_loss
            opt_nbf.zero_grad()
            loss.backward()
            opt_nbf.step()
            total += loss.item()
        print(f"nbf_epoch={epoch+1} loss={total / max(1, len(train_loader)):.6f}", flush=True)

    torch.save({
        "ssm": ssm.state_dict(),
        "nbf": nbf.state_dict(),
        "embedding_dim": dim,
        "ssm_hidden": args.ssm_hidden,
        "nbf_hidden": args.nbf_hidden,
        "embedding_model": args.embedding_model,
        "kappa": args.kappa,
        "guard_model_params_M": args.guard_model_params_m,
    }, out_dir / "nbf_model.pt")


def predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.model_path, map_location=device)
    embedder = SentenceTransformer(ckpt["embedding_model"], device="cpu")
    dataset = TrajectoryDataset([args.input_data], embedder, args.cache)
    dim = ckpt["embedding_dim"]
    ssm = NeuralStateSpaceModel(dim, ckpt["ssm_hidden"]).to(device)
    nbf = NeuralBarrierFunction(dim, ckpt["nbf_hidden"]).to(device)
    ssm.load_state_dict(ckpt["ssm"])
    nbf.load_state_dict(ckpt["nbf"])
    ssm.eval()
    nbf.eval()
    out = []
    with torch.no_grad():
        for row in dataset.encoded:
            x = torch.zeros(1, dim, device=device)
            turns = []
            for turn_idx in range(row["u"].shape[0]):
                u = row["u"][turn_idx].unsqueeze(0).to(device)
                logits = nbf(x, u)
                margin = barrier_margin(logits).item()
                pred = int(logits.argmax(-1).item()) + 1
                filtered = margin + args.threshold >= 0
                x, _ = ssm(x, u)
                turns.append({
                    "turn": turn_idx + 1,
                    "attack": row["attack"],
                    "true_score": int(row["labels"][turn_idx].item()) + 1,
                    "pred_score": pred,
                    "barrier_margin": margin,
                    "filtered": bool(filtered),
                })
            out.append({"id": row["id"], "split": row["split"], "attack": row["attack"], "turns": turns})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({
        "threshold": args.threshold,
        "guard_model_params_M": ckpt.get("guard_model_params_M"),
        "dialogues": out,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_train = sub.add_parser("train")
    p_train.add_argument("--train-data", default="data/local_dialogues/train.jsonl")
    p_train.add_argument("--embedding-model", default="data/downloads/all-MiniLM-L6-v2")
    p_train.add_argument("--cache", default="tmp/train_embeddings.pt")
    p_train.add_argument("--out-dir", default="checkpoints/nbf")
    p_train.add_argument("--ssm-hidden", type=int, default=256)
    p_train.add_argument("--nbf-hidden", type=int, default=32)
    p_train.add_argument("--ssm-epochs", type=int, default=16)
    p_train.add_argument("--nbf-epochs", type=int, default=24)
    p_train.add_argument("--batch-size", type=int, default=64)
    p_train.add_argument("--ssm-lr", type=float, default=1e-4)
    p_train.add_argument("--nbf-lr", type=float, default=1e-3)
    p_train.add_argument("--safe-weight", type=float, default=100.0)
    p_train.add_argument("--invariance-weight", type=float, default=100.0)
    p_train.add_argument("--train-eta", type=float, default=0.0)
    p_train.add_argument("--kappa", type=int, default=3)
    p_train.add_argument("--guard-model-params-m", type=float, default=23.1)
    p_train.add_argument("--seed", type=int, default=13)
    p_pred = sub.add_parser("predict")
    p_pred.add_argument("--model-path", default="checkpoints/nbf/nbf_model.pt")
    p_pred.add_argument("--input-data", required=True)
    p_pred.add_argument("--cache", default=None)
    p_pred.add_argument("--threshold", type=float, default=0.001)
    p_pred.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.cmd == "train":
        train(args)
    else:
        predict(args)

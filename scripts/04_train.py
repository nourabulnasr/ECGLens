"""Train one arm: `python scripts/04_train.py translated|untranslated [epochs]`.

Text embeddings are frozen MiniLM vectors (precomputed, per arm); the ECG encoder and a small text head are trained
with multi-positive symmetric InfoNCE on folds 1-8. Fold 9 picks the checkpoint (strict ECG->report R@10); fold 10
is never touched here.
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ecglens.data import TEST_FOLD, TRAIN_FOLDS, VAL_FOLD, build_tables
from ecglens.evaluation import embed_ecgs, prep, recall_at, strict_ranks
from ecglens.model import ECGLens, multi_positive_infonce

arm = sys.argv[1]
epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 20
SEED, BS, LR = 0, 256, 2e-3
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(int(os.environ.get("ECGLENS_THREADS", 6)))

df, unique_clean = build_tables()
ids = np.load("data/signals_ids.npy")
row_of = {e: i for i, e in enumerate(ids)}
df["row"] = df.index.map(row_of)
X = np.load("data/signals_100.npy", mmap_mode="r")
T = torch.from_numpy(np.load(f"data/text_emb_{arm}.npy"))

tr = df[df.fold.isin(TRAIN_FOLDS)]
va = df[df.fold == VAL_FOLD]
assert TEST_FOLD not in set(tr.fold) | set(va.fold)
tr_rows, tr_sid = tr.row.to_numpy(), torch.tensor(tr.sid.to_numpy())

model = ECGLens()
n_params = sum(p.numel() for p in model.ecg.parameters())
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
steps_per_epoch = len(tr) // BS
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=epochs * steps_per_epoch, pct_start=0.15)


def val_metrics():
    """Strict ECG->report retrieval on the val fold's unique-report gallery."""
    ze = embed_ecgs(model, X, va.row.to_numpy())
    g_sid = sorted(va.sid.unique())
    pos = {s: i for i, s in enumerate(g_sid)}
    with torch.no_grad():
        zg = model.text(T[g_sid]).numpy()
    ranks = strict_ranks(ze @ zg.T, va.sid.map(pos).to_numpy())
    return recall_at(ranks), len(g_sid)


os.makedirs("models", exist_ok=True)
log_path = f"models/train_log_{arm}.jsonl"
open(log_path, "w").close()
best = -1.0
for ep in range(epochs):
    model.train()
    t0 = time.time()
    perm = np.random.permutation(len(tr))
    losses = []
    for b in range(steps_per_epoch):
        idx = perm[b * BS : (b + 1) * BS]
        xb = prep(X[tr_rows[idx]])
        # light augmentation: global gain jitter, sensor noise, small circular time shift
        xb = xb * torch.empty(len(idx), 1, 1).uniform_(0.9, 1.1) + 0.02 * torch.randn_like(xb)
        xb = torch.roll(xb, int(np.random.randint(-40, 41)), dims=-1)
        sid = tr_sid[idx]
        loss = multi_positive_infonce(model.ecg(xb), model.text(T[sid]), sid, model.log_scale)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
    m, n_gal = val_metrics()
    rec = {"epoch": ep + 1, "train_loss": float(np.mean(losses)), "val": m, "val_gallery": n_gal,
           "sec_per_epoch": time.time() - t0, "ecg_params": n_params}
    print(json.dumps(rec), flush=True)
    open(log_path, "a").write(json.dumps(rec) + "\n")
    if m["R@10"] > best:
        best = m["R@10"]
        torch.save({"state": model.state_dict(), "epoch": ep + 1, "val_R@10": best, "arm": arm}, f"models/ecglens_{arm}.pt")
print("best val R@10", best)

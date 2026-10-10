"""Post-hoc diagnostic: reports that still contain marker remnants ('4.46', 'unbest...') or cardiologist 'Edit:' annotations
(which quote SCP codes, i.e. the labels). Compare retrieval on test queries WITH vs WITHOUT such text. Append-only to diagnostics.json."""
import datetime
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ecglens.data import TEST_FOLD, TRAIN_FOLDS, build_tables
from ecglens.evaluation import embed_ecgs, strict_ranks
from ecglens.model import ECGLens

df, _ = build_tables()
df["row"] = df.index.map({e: i for i, e in enumerate(np.load("data/signals_ids.npy"))})
X = np.load("data/signals_100.npy", mmap_mode="r")
te = df[df.fold == TEST_FOLD]
flag = lambda s: ("4.46" in s) or ("unbest" in s and "bericht" in s) or ("edit:" in s)  # noqa: E731
df["leaky"] = df.report_clean.map(flag)
te = df[df.fold == TEST_FOLD]
g_sid = sorted(te.sid.unique()); pos = {s: i for i, s in enumerate(g_sid)}
tc = te.sid.map(pos).to_numpy()
out = {"run_id": datetime.datetime.now().isoformat(timespec="seconds"), "kind": "marker_and_edit_annotation_subset",
       "frac_all_rows_with_remnant_or_edit": float(df.leaky.mean()), "frac_test_queries_with_remnant_or_edit": float(te.leaky.mean()),
       "n_test_leaky": int(te.leaky.sum()), "n_train_leaky": int(df[df.fold.isin(TRAIN_FOLDS)].leaky.sum())}
for arm in ("translated", "untranslated"):
    ck = torch.load(f"models/ecglens_{arm}.pt", weights_only=False)
    m = ECGLens(); m.load_state_dict(ck["state"]); m.eval()
    T = np.load(f"data/text_emb_{arm}.npy")
    ze = embed_ecgs(m, X, te.row.to_numpy())
    with torch.no_grad():
        zg = m.text(torch.from_numpy(T[g_sid])).numpy()
    hit = strict_ranks(ze @ zg.T, tc) <= 10
    lk = te.leaky.to_numpy()
    out[arm] = {"strict_R@10_all": float(hit.mean()), "strict_R@10_queries_without_remnant": float(hit[~lk].mean()),
                "strict_R@10_queries_with_remnant": float(hit[lk].mean()) if lk.any() else None}
p = "diagnostics.json"
prev = json.load(open(p))["runs"]; prev.append(out); json.dump({"runs": prev}, open(p, "w"), indent=1)
print(json.dumps(out, indent=1))

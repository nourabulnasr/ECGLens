"""ECGLens 30-second demo. No API keys, no PTB-XL download (uses the committed demo/ pack).

  python demo.py                      # 3 sample ECGs: top-5 retrieved reports + zero-shot triage vs the true labels
  python demo.py --query "atrial fibrillation with left bundle branch block"   # report -> most similar ECGs (needs MiniLM, ~90 MB, cached after 1st run)

Research retrieval demo, NOT a diagnostic device.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch

from ecglens.evaluation import prep
from ecglens.model import ECGLens

ap = argparse.ArgumentParser()
ap.add_argument("--query", default=None)
ap.add_argument("--n", type=int, default=3)
args = ap.parse_args()

pack = np.load("demo/pack.npz")
space = np.load("demo/text_space.npz")
gal = json.load(open("demo/gallery.json", encoding="utf-8"))
G, classes = gal["gallery"], gal["superclasses"]
pidx = json.load(open("data/prompts_index.json"))["index"] if os.path.exists("data/prompts_index.json") else json.load(open("demo/prompts_index.json"))["index"]
model = ECGLens()
model.load_state_dict(torch.load("demo/model.pt", weights_only=False)["state"])
model.eval()
with torch.no_grad():
    ze = model.ecg(prep(pack["ecg"])).numpy()
zg, zp = space["gallery"].astype(np.float32), space["prompts"].astype(np.float32)

if args.query is None:
    for i in range(args.n):
        idx = int(np.random.default_rng(i).integers(0, len(ze)))
        sim = zg @ ze[idx]
        top = np.argsort(-sim)[:5]
        true = pack["true_gallery_idx"][idx]
        print(f"\n=== ECG {int(pack['ecg_id'][idx])}  (held-out fold 10) ===")
        print("true report   :", G[true]["english"])
        print("true labels   :", [c for c, m in zip(classes, pack["superclass_mask"][idx]) if m] or ["(none)"])
        print("top-5 reports :")
        for r, j in enumerate(top, 1):
            print(f"  {r}. {sim[j]:+.3f}  {G[j]['english'][:110]}{'   <-- exact paired report' if j == true else ''}")
        zs = {c: float((ze[idx] @ zp[pidx[f'statements/{c}']].T).mean()) for c in classes}
        print("zero-shot score per superclass (mean cosine to SCP-statement prompts):", {c: round(v, 3) for c, v in zs.items()})
else:
    json.dump([args.query], open("data/_tmp_query.json", "w"))
    subprocess.run([sys.executable, "-m", "ecglens.text_worker", "embed", "data/_tmp_query.json", "data/_tmp_query.npy"], check=True)
    with torch.no_grad():
        zq = model.text(torch.from_numpy(np.load("data/_tmp_query.npy"))).numpy()[0]
    sim = ze @ zq
    print(f'\nQuery: "{args.query}"  -> top-5 of {len(ze)} held-out ECGs')
    for r, j in enumerate(np.argsort(-sim)[:5], 1):
        print(f"  {r}. {sim[j]:+.3f}  ECG {int(pack['ecg_id'][j])}  report: {G[pack['true_gallery_idx'][j]]['english'][:100]}")

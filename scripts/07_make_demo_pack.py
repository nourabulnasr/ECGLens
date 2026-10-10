"""Build the small committed demo pack (so the demo runs without the 1.8 GB PTB-XL download).

demo/pack.npz : 300 held-out (fold 10) ECGs as int16, their superclass labels, and the gallery index of their own report
demo/gallery.json : every unique fold-10 report (original + English) -- the retrieval gallery
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ecglens.data import SUPERCLASSES, TEST_FOLD, build_tables
from ecglens.model import ECGLens

df, _ = build_tables()
ids = np.load("data/signals_ids.npy")
df["row"] = df.index.map({e: i for i, e in enumerate(ids)})
X = np.load("data/signals_100.npy", mmap_mode="r")
te = df[df.fold == TEST_FOLD]
g_sid = sorted(te.sid.unique())
pos = {s: i for i, s in enumerate(g_sid)}
first = te.drop_duplicates("sid").set_index("sid")
gallery = [{"original": first.loc[s, "report_clean"], "english": first.loc[s, "text_en"]} for s in g_sid]
demo = te.sample(300, random_state=0)
os.makedirs("demo", exist_ok=True)
np.savez_compressed(
    "demo/pack.npz",
    ecg=np.stack([X[r] for r in demo.row.to_numpy()]),
    true_gallery_idx=demo.sid.map(pos).to_numpy(),
    superclass_mask=np.stack(demo.y.to_numpy()).astype(np.int8),
    ecg_id=demo.index.to_numpy(),
)
json.dump({"gallery": gallery, "superclasses": SUPERCLASSES}, open("demo/gallery.json", "w", encoding="utf-8"), ensure_ascii=False)

# precompute gallery + prompt embeddings in the trained joint space (translated arm)
ck = torch.load("models/ecglens_translated.pt", weights_only=False)
m = ECGLens(); m.load_state_dict(ck["state"]); m.eval()
T = np.load("data/text_emb_translated.npy")
P = np.load("data/text_emb_prompts.npy")
with torch.no_grad():
    zg = m.text(torch.from_numpy(T[g_sid])).numpy()
    zp = m.text(torch.from_numpy(P)).numpy()
np.savez("demo/text_space.npz", gallery=zg.astype(np.float16), prompts=zp.astype(np.float16))
torch.save({"state": {k: v for k, v in m.state_dict().items()}, "arm": "translated"}, "demo/model.pt")
print("gallery", len(gallery), "demo ECGs", len(demo))

"""Post-hoc diagnostics (NOT part of the pre-registered claims; written to diagnostics.json, append-only).

Question: German and English reports likely come from different sites/devices, so a waveform encoder could be
retrieving report *language/style* rather than pathology. Test: (a) how often is the top-10 in the query's own
language vs the gallery base rate, (b) re-score retrieval with the gallery restricted to the query's own language.
"""
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
te, tr = df[df.fold == TEST_FOLD], df[df.fold.isin(TRAIN_FOLDS)]
g_sid = sorted(te.sid.unique()); pos = {s: i for i, s in enumerate(g_sid)}
true_col = te.sid.map(pos).to_numpy()
lang_of_sid = te.drop_duplicates("sid").set_index("sid").lang
g_lang = np.array([lang_of_sid[s] for s in g_sid]); q_lang = te.lang.to_numpy()
prior = np.array([tr.sid.value_counts().get(s, 0) for s in g_sid], dtype=float) + np.random.default_rng(0).random(len(g_sid)) * 1e-3
sets = te.groupby("sid").superclasses.agg(lambda s: s.value_counts().index[0])
g_set = np.array([hash(sets[s]) for s in g_sid]); q_set = np.array([hash(s) for s in te.superclasses])
out = {"run_id": datetime.datetime.now().isoformat(timespec="seconds"), "gallery_language_counts": {l: int((g_lang == l).sum()) for l in ("de", "en")},
       "test_query_language_counts": {l: int((q_lang == l).sum()) for l in ("de", "en")}}
for arm in ("translated", "untranslated"):
    ck = torch.load(f"models/ecglens_{arm}.pt", weights_only=False)
    m = ECGLens(); m.load_state_dict(ck["state"]); m.eval()
    T = np.load(f"data/text_emb_{arm}.npy")
    ze = embed_ecgs(m, X, te.row.to_numpy())
    with torch.no_grad():
        zg = m.text(torch.from_numpy(T[g_sid])).numpy()
    sim = ze @ zg.T
    top = np.argsort(-sim, axis=1)[:, :10]
    same_lang_top10 = float((g_lang[top] == q_lang[:, None]).mean())
    base_rate = float(np.mean([(g_lang == l).mean() for l in q_lang]))
    # restricted gallery: only reports in the query's own language
    r10, pri10, rnd10, p10, pri_p10 = [], [], [], [], []
    for l in ("de", "en"):
        qi = np.where(q_lang == l)[0]; gi = np.where(g_lang == l)[0]
        sub = sim[np.ix_(qi, gi)]
        gpos = {g: k for k, g in enumerate(gi)}
        tc = np.array([gpos[c] for c in true_col[qi]])
        rk = strict_ranks(sub, tc)
        r10 += list(rk <= 10)
        pri_rk = strict_ranks(np.tile(prior[gi], (len(qi), 1)), tc)
        pri10 += list(pri_rk <= 10)
        rnd10 += [min(10, len(gi)) / len(gi)] * len(qi)
        tp = np.argsort(-sub, axis=1)[:, :10]
        p10 += list((g_set[gi][tp] == q_set[qi][:, None]).mean(1))
        pp = np.argsort(-prior[gi])[:10]
        pri_p10 += list((g_set[gi][pp][None, :] == q_set[qi][:, None]).mean(1))
    out[arm] = {"top10_in_query_language": same_lang_top10, "top10_language_base_rate_if_random": base_rate,
                "same_language_gallery": {"strict_R@10": float(np.mean(r10)), "prior_strict_R@10": float(np.mean(pri10)),
                                           "random_strict_R@10": float(np.mean(rnd10)), "P@10_same_set": float(np.mean(p10)),
                                           "prior_P@10_same_set": float(np.mean(pri_p10))}}
    # does the ECG embedding alone predict report language? (site/device confound probe)
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    ftr = embed_ecgs(m, X, tr.row.to_numpy()); ytr = (tr.lang == "de").to_numpy()
    out[arm]["ecg_embedding_predicts_report_language_auroc"] = float(roc_auc_score((te.lang == "de").to_numpy(), LogisticRegression(max_iter=2000).fit(ftr, ytr).decision_function(ze)))
path = "diagnostics.json"
prev = json.load(open(path))["runs"] if os.path.exists(path) else []
prev.append(out)
json.dump({"runs": prev}, open(path, "w"), indent=1)
print(json.dumps(out, indent=1))

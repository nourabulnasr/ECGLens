"""EVAL GATE for ECGLens. Usage: `python scripts/05_eval.py [tag]`

Thresholds and the verdict rule below were written BEFORE any model was trained or evaluated.
Results are APPENDED to eval_results.json (never overwritten); the first run is ground truth.
"""
import datetime
import json
import os
import sys
import time

import faiss
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ecglens.data import SUPERCLASSES, TEST_FOLD, TRAIN_FOLDS, build_tables
from ecglens.evaluation import embed_ecgs, label_precision_at, prep, recall_at, strict_ranks
from ecglens.model import ECGLens

CLAIM_UNDER_TEST = (
    "A small 1D-ResNet ECG encoder contrastively aligned to sentence embeddings of PTB-XL reports "
    "(German reports machine-translated to English) supports ECG->report retrieval and zero-shot "
    "triage on the held-out fold 10: "
    "C1 (retrieval) strict ECG->report R@10 over the unique-report gallery is >= 5x random AND beats a "
    "training-frequency-prior baseline by >= 0.05 absolute on BOTH strict R@10 and same-superclass-set P@10; "
    "C2 (zero-shot) macro-AUROC over the 5 diagnostic superclasses with SCP-statement text prompts and no "
    "classifier head is >= 0.75; "
    "C3 (translation is load-bearing) translated-arm same-superclass-set P@10 exceeds the untranslated-arm "
    "value by >= 0.02 with a paired-bootstrap 95% CI lower bound > 0."
)
THRESH = {"R10_x_random": 5.0, "margin_over_prior": 0.05, "zs_macro_auc": 0.75, "translation_delta": 0.02}
VERDICT_RULE = "REFUTED if C1 fails; CONFIRMED if C1, C2 and C3 all pass; otherwise PARTIAL (say which failed)."
TAG = sys.argv[1] if len(sys.argv) > 1 else "run"
rng = np.random.default_rng(0)
torch.set_num_threads(6)

df, unique_clean = build_tables()
ids = np.load("data/signals_ids.npy")
df["row"] = df.index.map({e: i for i, e in enumerate(ids)})
X = np.load("data/signals_100.npy", mmap_mode="r")
te, tr = df[df.fold == TEST_FOLD], df[df.fold.isin(TRAIN_FOLDS)]
TEXT = {a: np.load(f"data/text_emb_{a}.npy") for a in ("translated", "untranslated")}
P = np.load("data/text_emb_prompts.npy")
pidx = json.load(open("data/prompts_index.json"))["index"]

# ---- gallery: unique report strings of the test fold. Exactly one strict-relevant item per query -> no ties.
g_sid = sorted(te.sid.unique())
pos = {s: i for i, s in enumerate(g_sid)}
true_col = te.sid.map(pos).to_numpy()
G = len(g_sid)
sets_by_sid = te.groupby("sid").superclasses.agg(lambda s: s.value_counts().index[0])  # modal set per string
gal_set = [sets_by_sid[s] for s in g_sid]
q_set = list(te.superclasses)
print(f"test ECGs {len(te)}, gallery unique reports {G}", flush=True)

# ---- baselines
random_R = {f"R@{k}": k / G for k in (1, 5, 10)}
random_P10 = float(np.mean([np.mean([gs == qs for gs in gal_set]) for qs in q_set]))
train_freq = tr.sid.value_counts()
prior_scores = np.array([train_freq.get(s, 0) for s in g_sid], dtype=np.float64) + rng.random(G) * 1e-3
prior_sim = np.tile(prior_scores, (len(te), 1))
prior_ranks = strict_ranks(prior_sim, true_col)
prior = {**recall_at(prior_ranks), "P@10_same_set": label_precision_at(prior_sim, q_set, gal_set)}
# labels-only oracle: rank gallery by superclass-set overlap with the query's TRUE labels, random tie-break
def jacc(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(len(a | b), 1)
oracle_sim = np.array([[jacc(qs, gs) for gs in gal_set] for qs in q_set]) + rng.random((len(te), G)) * 1e-4
oracle = {**recall_at(strict_ranks(oracle_sim, true_col)), "P@10_same_set": label_precision_at(oracle_sim, q_set, gal_set)}


def boot_ci(vals, n=1000):
    vals = np.asarray(vals, dtype=float)
    m = np.array([vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(n)])
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def per_query_p10(sim):
    top = np.argsort(-sim, axis=1)[:, :10]
    g = np.array([hash(s) for s in gal_set])
    q = np.array([hash(s) for s in q_set])
    return (g[top] == q[:, None]).mean(1)


arms, per_q = {}, {}
Xtr_feat, Xte_feat = None, None
for arm in ("translated", "untranslated"):
    ck = torch.load(f"models/ecglens_{arm}.pt", weights_only=False)
    model = ECGLens()
    model.load_state_dict(ck["state"])
    model.eval()
    ze = embed_ecgs(model, X, te.row.to_numpy())
    with torch.no_grad():
        zg = model.text(torch.from_numpy(TEXT[arm][g_sid])).numpy()
        zp = model.text(torch.from_numpy(P)).numpy()
    sim = ze @ zg.T
    # sanity: FAISS inner-product search must agree with the numpy ranking (top-10 as sets)
    index = faiss.IndexFlatIP(zg.shape[1])
    index.add(zg.astype(np.float32))
    _, I = index.search(ze.astype(np.float32), 10)
    faiss_agree = float(np.mean([set(I[i]) == set(np.argsort(-sim[i])[:10]) for i in range(len(te))]))
    ranks = strict_ranks(sim, true_col)
    p10_q = per_query_p10(sim)
    per_q[arm] = {"p10": p10_q, "r10": (ranks <= 10).astype(float)}

    # report -> ECG (secondary): each unique test report is a query over all test ECGs
    sim_t2e = zg @ ze.T
    top = np.argsort(-sim_t2e, axis=1)[:, :10]
    t2e_hit = np.mean([(te.sid.to_numpy()[top[i]] == g_sid[i]).any() for i in range(G)])
    gs_hash = np.array([hash(s) for s in gal_set])
    te_hash = np.array([hash(s) for s in q_set])
    t2e_p10 = float((te_hash[top] == gs_hash[:, None]).mean())
    counts = te.sid.value_counts()
    N = len(te)
    from math import comb
    t2e_rand = float(np.mean([1 - comb(N - counts[s], 10) / comb(N, 10) for s in g_sid]))

    # zero-shot: score = mean cosine to the prompt set of each superclass; AUROC over ECGs with >=1 diagnostic label
    lab = te[te.superclasses.map(len) > 0]
    lab_idx = np.array([te.index.get_loc(i) for i in lab.index])
    Y = np.stack(lab.y.to_numpy())
    zs = {}
    for kind in ("statements", "names"):
        aucs = {}
        for ci, c in enumerate(SUPERCLASSES):
            sc = (ze[lab_idx] @ zp[pidx[f"{kind}/{c}"]].T).mean(1)
            aucs[c] = float(roc_auc_score(Y[:, ci], sc))
        zs[kind] = {"per_class": aucs, "macro": float(np.mean(list(aucs.values())))}

    arms[arm] = {
        "best_epoch": ck["epoch"], "val_R@10_at_best": ck["val_R@10"],
        "ecg_to_report_strict": recall_at(ranks),
        "ecg_to_report_R@10_ci95": boot_ci(per_q[arm]["r10"]),
        "ecg_to_report_P@10_same_set": float(p10_q.mean()),
        "ecg_to_report_P@10_same_set_ci95": boot_ci(p10_q),
        "report_to_ecg": {"R@10_any_identical_report": float(t2e_hit), "R@10_random": t2e_rand, "P@10_same_set": t2e_p10},
        "zero_shot_auroc": zs,
        "faiss_top10_agrees_with_numpy": faiss_agree,
    }
    if arm == "translated":
        tmodel, tze, tzg = model, ze, zg
        Xte_feat = embed_ecgs(model, X, te.row.to_numpy(), features=True)
        Xtr_feat = embed_ecgs(model, X, tr.row.to_numpy(), features=True)
        Ytr_all, tr_lab = np.stack(tr.y.to_numpy()), tr.superclasses.map(len).to_numpy() > 0

# ---- paired bootstrap on the translation ablation (same test queries, same gallery)
d = per_q["translated"]["p10"] - per_q["untranslated"]["p10"]
delta = {"P@10_same_set_translated_minus_untranslated": float(d.mean()), "ci95": boot_ci(d)}
d_r = per_q["translated"]["r10"] - per_q["untranslated"]["r10"]
delta["strict_R@10_translated_minus_untranslated"] = float(d_r.mean())
delta["strict_R@10_ci95"] = boot_ci(d_r)
delta["zero_shot_macro_translated_minus_untranslated"] = (
    arms["translated"]["zero_shot_auroc"]["statements"]["macro"] - arms["untranslated"]["zero_shot_auroc"]["statements"]["macro"]
)

# ---- supervised linear probe on the SAME frozen encoder (translated arm), for the zero-shot comparison
lab = te[te.superclasses.map(len) > 0]
lab_idx = np.array([te.index.get_loc(i) for i in lab.index])
Yte = np.stack(lab.y.to_numpy())
sup = {}
for ci, c in enumerate(SUPERCLASSES):
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr_feat[tr_lab], Ytr_all[tr_lab, ci])
    sup[c] = float(roc_auc_score(Yte[:, ci], clf.decision_function(Xte_feat[lab_idx])))
supervised = {"per_class": sup, "macro": float(np.mean(list(sup.values())))}

# ---- label-leakage quantification: how much do report TEXT alone tell you about the superclass labels?
Ttr = TEXT["translated"][tr.sid.to_numpy()][tr_lab]
Tte = TEXT["translated"][te.sid.to_numpy()][lab_idx]
leak = {}
for ci, c in enumerate(SUPERCLASSES):
    clf = LogisticRegression(max_iter=2000).fit(Ttr, Ytr_all[tr_lab, ci])
    leak[c] = float(roc_auc_score(Yte[:, ci], clf.decision_function(Tte)))
modal = tr.groupby("sid").superclasses.agg(lambda s: s.value_counts().index[0])
seen = te.sid.isin(modal.index)
purity = float(np.mean([modal[s] == qs for s, qs in zip(te.sid[seen], te.superclasses[seen])]))
leakage = {"report_text_only_probe_auroc": {"per_class": leak, "macro": float(np.mean(list(leak.values())))},
           "report_string_predicts_own_superclass_set_acc": purity, "test_reports_seen_in_train_frac": float(seen.mean())}

# ---- latency / size
lat = {}
xb = prep(np.asarray(X[te.row.to_numpy()[:1]]))
torch.set_num_threads(1)
tmodel.eval()
ts = []
with torch.no_grad():
    for _ in range(100):
        t0 = time.perf_counter(); tmodel.ecg(xb); ts.append(time.perf_counter() - t0)
lat["single_ecg_encode_ms_1thread_median"] = float(np.median(ts) * 1e3)
full_idx = faiss.IndexFlatIP(128)
with torch.no_grad():
    full_g = tmodel.text(torch.from_numpy(TEXT["translated"])).numpy().astype(np.float32)
full_idx.add(full_g)
ts = []
for i in range(200):
    q = tze[i : i + 1].astype(np.float32); t0 = time.perf_counter(); full_idx.search(q, 10); ts.append(time.perf_counter() - t0)
lat["faiss_query_ms_median_over_all_%d_unique_reports" % len(full_g)] = float(np.median(ts) * 1e3)
log = [json.loads(l) for l in open("models/train_log_translated.jsonl")]
lat["train_sec_per_epoch_mean"] = float(np.mean([r["sec_per_epoch"] for r in log]))
lat["ecg_encoder_params"] = int(log[0]["ecg_params"])

# ---- claim checks (thresholds fixed above, before running)
tr_arm = arms["translated"]
R10 = tr_arm["ecg_to_report_strict"]["R@10"]
c1 = {
    "R10_vs_random": {"value": R10, "random": random_R["R@10"], "ratio": R10 / random_R["R@10"], "pass": R10 >= THRESH["R10_x_random"] * random_R["R@10"]},
    "R10_vs_prior": {"value": R10, "prior": prior["R@10"], "pass": R10 >= prior["R@10"] + THRESH["margin_over_prior"]},
    "P10_vs_prior": {"value": tr_arm["ecg_to_report_P@10_same_set"], "prior": prior["P@10_same_set"],
                     "pass": tr_arm["ecg_to_report_P@10_same_set"] >= prior["P@10_same_set"] + THRESH["margin_over_prior"]},
}
c1["pass"] = all(v["pass"] for v in c1.values())
zs_macro = tr_arm["zero_shot_auroc"]["statements"]["macro"]
c2 = {"value": zs_macro, "threshold": THRESH["zs_macro_auc"], "pass": zs_macro >= THRESH["zs_macro_auc"]}
c3 = {"value": delta["P@10_same_set_translated_minus_untranslated"], "ci95": delta["ci95"],
      "pass": delta["P@10_same_set_translated_minus_untranslated"] >= THRESH["translation_delta"] and delta["ci95"][0] > 0}
verdict = "REFUTED" if not c1["pass"] else ("CONFIRMED" if (c2["pass"] and c3["pass"]) else "PARTIAL")

run = {
    "run_id": f"{TAG}-{datetime.datetime.now().isoformat(timespec='seconds')}",
    "CLAIM_UNDER_TEST": CLAIM_UNDER_TEST, "thresholds_preregistered": THRESH, "verdict_rule": VERDICT_RULE,
    "setup": {"test_ecgs": int(len(te)), "gallery_unique_reports": G, "train_ecgs": int(len(tr)),
              "note": "strict relevance = identical cleaned report string; same-set relevance = same PTB-XL diagnostic-superclass set"},
    "baselines": {"random": {**random_R, "P@10_same_set": random_P10}, "training_frequency_prior": prior, "labels_only_oracle_random_tiebreak": oracle},
    "arms": arms, "translation_ablation_delta": delta,
    "supervised_linear_probe_same_encoder": supervised, "label_leakage": leakage, "latency_and_size": lat,
    "claim_checks": {"C1_retrieval": c1, "C2_zero_shot": c2, "C3_translation_load_bearing": c3}, "verdict": verdict,
}
path = "eval_results.json"
runs = json.load(open(path))["runs"] if os.path.exists(path) else []  # append-only
runs.append(run)
json.dump({"runs": runs}, open(path, "w"), indent=1)
print(json.dumps({k: run[k] for k in ("baselines", "translation_ablation_delta", "supervised_linear_probe_same_encoder", "label_leakage", "latency_and_size", "claim_checks", "verdict")}, indent=1))
print("arms:", json.dumps({a: {k: v for k, v in arms[a].items() if k != "zero_shot_auroc"} for a in arms}, indent=1))
print("zero-shot:", json.dumps({a: arms[a]["zero_shot_auroc"] for a in arms}, indent=1))

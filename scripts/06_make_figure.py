"""Render docs/results.png from eval_results.json (no hand-typed numbers).

Usage: python scripts/06_make_figure.py [run_index]   (default: the latest run; index 0 = the first, ground-truth run)
Palette: blue = translated arm, orange = untranslated arm, gray = reference baselines (validated for CVD separation).
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, GRAY, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#898781", "#0b0b0b", "#52514e", "#e1e0d9"
runs = json.load(open("eval_results.json"))["runs"]
i = int(sys.argv[1]) if len(sys.argv) > 1 else len(runs) - 1
r = runs[i]
A, B = r["arms"]["translated"], r["arms"]["untranslated"]
base = r["baselines"]

plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), facecolor="#fcfcfb")
for a in ax:
    a.set_facecolor("#fcfcfb")
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    a.yaxis.grid(True, color=GRID, lw=0.8)
    a.set_axisbelow(True)


def bars(a, labels, vals, colors, fmt="{:.3f}"):
    x = range(len(vals))
    a.bar(x, vals, color=colors, width=0.62)
    for xi, v in zip(x, vals):
        a.text(xi, v + 0.012, fmt.format(v), ha="center", va="bottom", color=INK, fontsize=9)
    a.set_xticks(list(x))
    a.set_xticklabels(labels, fontsize=9)
    a.set_ylim(0, min(1.0, max(vals) * 1.22 + 0.02))


lab = ["random", "frequency\nprior", "untranslated\nDE text", "translated\nEN text"]
col = [GRAY, GRAY, ORANGE, BLUE]
bars(ax[0], lab, [base["random"]["R@10"], base["training_frequency_prior"]["R@10"], B["ecg_to_report_strict"]["R@10"], A["ecg_to_report_strict"]["R@10"]], col)
ax[0].set_title(f"ECG -> exact report, Recall@10\n(fold 10, {r['setup']['gallery_unique_reports']} unique reports)", color=INK, fontsize=10.5, loc="left")

bars(ax[1], lab, [base["random"]["P@10_same_set"], base["training_frequency_prior"]["P@10_same_set"], B["ecg_to_report_P@10_same_set"], A["ecg_to_report_P@10_same_set"]], col)
ax[1].set_title("ECG -> report, Precision@10\n(retrieved report has the same diagnostic-superclass set)", color=INK, fontsize=10.5, loc="left")

classes = list(A["zero_shot_auroc"]["statements"]["per_class"])
w = 0.27
zs = [A["zero_shot_auroc"]["statements"]["per_class"][c] for c in classes]
zsu = [B["zero_shot_auroc"]["statements"]["per_class"][c] for c in classes]
sup = [r["supervised_linear_probe_same_encoder"]["per_class"][c] for c in classes]
for k, (vals, colr, name) in enumerate([(zs, BLUE, "zero-shot, translated"), (zsu, ORANGE, "zero-shot, untranslated"), (sup, GRAY, "supervised linear probe")]):
    ax[2].bar([j + (k - 1) * w for j in range(5)], vals, w * 0.92, color=colr, label=name)
ax[2].axhline(0.5, color=MUTED, lw=1, ls=(0, (4, 3)))
ax[2].text(-0.45, 0.505, "chance", color=MUTED, fontsize=8, ha="left", va="bottom")
ax[2].set_xticks(range(5))
ax[2].set_xticklabels(classes)
ax[2].set_ylim(0.4, 1.12)
ax[2].set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])  # AUROC cannot exceed 1; headroom above is for the legend
ax[2].set_title(f"Zero-shot triage AUROC per superclass (no classifier head)\nmacro: translated {A['zero_shot_auroc']['statements']['macro']:.3f} / untranslated {B['zero_shot_auroc']['statements']['macro']:.3f}\n/ supervised probe {r['supervised_linear_probe_same_encoder']['macro']:.3f}", color=INK, fontsize=10, loc="left")
ax[2].legend(frameon=False, fontsize=8, loc="upper center", ncol=3, columnspacing=1.0, handlelength=1.2)
fig.suptitle(f"ECGLens on PTB-XL held-out fold 10  |  run {r['run_id']}  |  verdict: {r['verdict']}", color=INK, fontsize=11.5, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.94))
os.makedirs("docs", exist_ok=True)
fig.savefig("docs/results.png", dpi=140, facecolor=fig.get_facecolor())
print("wrote docs/results.png from run", r["run_id"])

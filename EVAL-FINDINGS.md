# EVAL-FINDINGS - ECGLens

**Final verdict: PARTIAL**

## CLAIM_UNDER_TEST (stated before any model was trained; verbatim from `scripts/05_eval.py`)

A small 1D-ResNet ECG encoder contrastively aligned to sentence embeddings of PTB-XL reports (German reports machine-translated to English)
supports ECG->report retrieval and zero-shot triage on the held-out fold 10:

* **C1 (retrieval):** strict ECG->report R@10 over the unique-report gallery is >= 5x random AND beats a training-frequency-prior baseline by >= 0.05 absolute on BOTH strict R@10 and same-superclass-set P@10.
* **C2 (zero-shot):** macro-AUROC over the 5 diagnostic superclasses, with SCP-statement text prompts and no classifier head, is >= 0.75.
* **C3 (translation is load-bearing):** translated-arm same-superclass-set P@10 exceeds the untranslated-arm value by >= 0.02 with a paired-bootstrap 95% CI lower bound > 0.

Verdict rule (also pre-registered): REFUTED if C1 fails; CONFIRMED if C1, C2 and C3 all pass; otherwise PARTIAL.

Setup: 21,215 records (584 Swedish-flagged excluded), train folds 1-8 (17,065 ECGs), checkpoint selection on fold 9, test = fold 10 (2,090 ECGs, 1,237 unique reports as the gallery).
Both arms: identical code, seed 0, 20 epochs, only the frozen text embeddings differ (translated English vs original German/English).

## Runs, in order

### Run 1 - `first-run-2026-09-25T23:28:13` (ground truth, stored as run 0 in `eval_results.json`)

| Metric (fold 10) | random | freq. prior | untranslated | translated |
|---|---|---|---|---|
| strict ECG->report R@1 / R@5 / R@10 | 0.001 / 0.004 / 0.008 | 0.090 / 0.259 / 0.288 | 0.106 / 0.312 / 0.445 | 0.100 / 0.315 / **0.439** |
| strict median rank / MRR | - | 227.5 / 0.164 | 13 / 0.215 | 14 / 0.208 |
| same-superclass-set P@10 | 0.141 | 0.355 | 0.425 | **0.437** |
| zero-shot macro-AUROC (statement prompts) | - | - | 0.658 | **0.751** |
| zero-shot macro-AUROC (short-name prompts) | - | - | 0.740 | 0.733 |

Supervised linear probe on the same frozen (translated-arm) encoder: macro-AUROC 0.915 (NORM .941, MI .914, STTC .933, CD .907, HYP .878).
Zero-shot per class, translated / statement prompts: NORM .875, MI .661, STTC .709, CD .774, HYP .735.
Labels-only oracle (ranks by true superclass set, random tie-break): strict R@10 0.092, P@10 0.979 (sanity reference).

Claim checks:
* **C1 PASS** - R@10 0.439 = 54x random (needed 5x); R@10 0.439 vs prior 0.288 (+0.151); P@10 0.437 vs prior 0.355 (+0.081). 95% bootstrap CI for R@10: 0.417-0.460.
* **C2 PASS, margin 0.0009** - 0.7509 vs threshold 0.75. This is a coin-flip-thin pass: the pre-registered prompt set clears the bar, the alternative short-name prompt set (0.733) does not, and there is a single seed.
* **C3 FAIL** - translated minus untranslated same-set P@10 = +0.0123, 95% CI [0.0050, 0.0199]. The effect is statistically distinguishable from zero but below the pre-registered +0.02 bar. Strict R@10 delta = -0.0062 (CI [-0.027, 0.012], i.e. no difference). Zero-shot macro-AUROC delta = +0.092 (statement prompts) but -0.007 with short-name prompts.

**Verdict by the pre-registered rule: PARTIAL** (C1 pass, C2 pass, C3 fail).

### Fix attempts: none made (0 of 2 used) - deliberately

* The one failed claim is C3, a narrow miss (+0.012 vs +0.02). No bug, wrong scale, or broken data path was found that would explain it, and the only available "fix" would have been lowering the +0.02 threshold - which is moving the goalposts, so it was not done.
* Diagnosis of *why* C3 misses (hypotheses, supported by data, not "fixes"): (a) exact-report identification only needs a distinguishable embedding, so untranslated German through MiniLM is as good as translated (strict R@10 0.445 vs 0.439); (b) MarianMT garbles clinical shorthand ("Vorhofflimmern" -> "pre-throat fibrillation"), capping the semantic benefit; (c) the translation benefit shows up in *semantic* tasks (zero-shot with statement prompts, +0.09) but not consistently (short-name prompts flip sign).

## Post-hoc diagnostics (NOT part of the pre-registered claims; `diagnostics.json`, appended, run after seeing run 1)

1. **Site/language confound.** German and English reports likely come from different recording sites. A logistic regression on the ECG embedding alone predicts report language with AUROC 0.860 (translated arm) / 0.859 (untranslated). Top-10 retrieved reports share the query's language 73.4% / 74.2% of the time vs a 53.2% base rate. **Language-restricted gallery** (the style cue removed): translated strict R@10 0.508 vs prior 0.313 vs random 0.016; same-set P@10 0.435 vs prior 0.334. **Reading:** part of the headline retrieval is site/style, but the model still clearly beats the prior within a language, so C1 survives this control.
2. **Label leakage in the text.** A text-only probe (report embedding -> superclass) reaches macro-AUROC 0.927; a report string (seen in training) predicts its ECG's superclass set 79.7% of the time (55.6% of test reports are seen in training). The retrieval task therefore partly reduces to label prediction, as expected since the SCP codes derive from the reports.
3. **Marker/annotation remnants.** 3.6% of all rows (5.0% of test queries, n=104) still contain `4.46 ... unbestaetigter bericht` remnants or cardiologist `Edit: ...` annotations that quote labels. Test strict R@10 without those queries: 0.433 (translated) / 0.437 (untranslated) vs 0.439 / 0.445 overall. These rows were **not** removed from training (discovered after the first run; retraining was not justified for <4% of rows and was not a diagnosis of the C3 failure).

## Other measured facts
Encoder 1,149,824 params; 122.6 s/epoch mean (12-core CPU, shared); single-ECG encode 6.2 ms (1 thread, median); FAISS flat-IP query 0.41 ms over all 9,237 unique reports; FAISS top-10 agreed with the exact numpy top-10 for 99.0% of translated-arm queries (ties/float rounding).
Report -> ECG (secondary, no pre-registered threshold): translated R@10 (any identical-report ECG in top-10) 0.494 vs 0.0075 random; untranslated 0.508.

## Final verdict: PARTIAL
Works: cross-modal ECG<->report retrieval far above chance and above a frequency prior (also within a single language), and zero-shot superclass scoring that just clears 0.75 with statement prompts.
Fails: the "translation is load-bearing" claim (+0.012 vs +0.02 needed; nil on exact-report retrieval; sign-unstable on zero-shot). Caveats: single seed, ECG embeddings carry a recording-site fingerprint, label leakage through report text is high, and zero-shot is well below a supervised probe (0.751 vs 0.915).

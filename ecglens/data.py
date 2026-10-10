"""Tables, splits and labels for PTB-XL. Everything derives from committed/regenerable files in data/."""
import ast
import json

import numpy as np
import pandas as pd

from .textprep import load_database

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
TRAIN_FOLDS, VAL_FOLD, TEST_FOLD = list(range(1, 9)), 9, 10  # PTB-XL recommended split


def build_tables(raw_dir: str = "data/raw", trans_path: str = "data/translations_de_en.json", with_text: bool = True):
    """Return (df, unique_clean, sid_of_row).

    df has: report_clean, lang, text_en (English text used by the 'translated' arm), superclasses (tuple),
            y (5-dim multi-hot), fold, sid (index into unique_clean). Swedish-flagged rows are excluded.
    """
    df = load_database(f"{raw_dir}/ptbxl_database.csv")
    df = df[df.lang != "sv"].copy()
    if with_text:  # translation cache is only needed for text-side work, not for packing waveforms
        trans = json.load(open(trans_path, encoding="utf-8"))
        df["text_en"] = [trans[t] if lang == "de" else t for t, lang in zip(df.report_clean, df.lang)]

    stm = pd.read_csv(f"{raw_dir}/scp_statements.csv", index_col=0)
    diag = stm[stm.diagnostic == 1].diagnostic_class.to_dict()  # SCP code -> superclass

    def supers(codes: str):
        found = {diag[c] for c in ast.literal_eval(codes) if c in diag}
        return tuple(s for s in SUPERCLASSES if s in found)

    df["superclasses"] = df.scp_codes.map(supers)
    df["y"] = df.superclasses.map(lambda s: np.array([c in s for c in SUPERCLASSES], dtype=np.float32))
    df["fold"] = df.strat_fold
    unique_clean = sorted(df.report_clean.unique())
    sid = {t: i for i, t in enumerate(unique_clean)}
    df["sid"] = df.report_clean.map(sid)
    return df, unique_clean


def scp_prompts(raw_dir: str = "data/raw"):
    """Zero-shot prompt sets per superclass, built from the SCP-ECG statement table (no labels involved).

    'statements': description of every diagnostic SCP statement in the superclass (pre-registered primary).
    'names'     : one short name per superclass (secondary).
    """
    stm = pd.read_csv(f"{raw_dir}/scp_statements.csv", index_col=0)
    d = stm[stm.diagnostic == 1]
    statements = {c: [str(x).lower() for x in d[d.diagnostic_class == c].description] for c in SUPERCLASSES}
    names = {
        "NORM": ["normal ecg"],
        "MI": ["myocardial infarction"],
        "STTC": ["st-t changes"],
        "CD": ["conduction disturbance"],
        "HYP": ["hypertrophy"],
    }
    return {"statements": statements, "names": names}

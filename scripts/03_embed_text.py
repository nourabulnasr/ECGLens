"""MiniLM sentence embeddings for every unique report string, in both ablation arms, plus zero-shot prompts.

  translated arm   : embed(English text; German reports machine-translated)
  untranslated arm : embed(original cleaned text, German left as German)
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ecglens.data import build_tables, scp_prompts
from ecglens.text_worker import run_worker

df, unique_clean = build_tables()
trans = json.load(open("data/translations_de_en.json", encoding="utf-8"))
lang = df.drop_duplicates("report_clean").set_index("report_clean").lang
en_text = [trans[t] if lang[t] == "de" else t for t in unique_clean]

json.dump(en_text, open("data/_tmp_en.json", "w", encoding="utf-8"), ensure_ascii=False)
json.dump(unique_clean, open("data/_tmp_orig.json", "w", encoding="utf-8"), ensure_ascii=False)
run_worker("embed", "data/_tmp_en.json", "data/text_emb_translated.npy")
run_worker("embed", "data/_tmp_orig.json", "data/text_emb_untranslated.npy")

prompts = scp_prompts()
flat, index = [], {}
for kind, d in prompts.items():
    for cls, plist in d.items():
        index[f"{kind}/{cls}"] = [len(flat) + i for i in range(len(plist))]
        flat += plist
json.dump(flat, open("data/_tmp_prompts.json", "w", encoding="utf-8"))
run_worker("embed", "data/_tmp_prompts.json", "data/text_emb_prompts.npy")
json.dump({"index": index, "prompts": flat}, open("data/prompts_index.json", "w"), indent=1)
print({k: len(v) for k, v in index.items()})

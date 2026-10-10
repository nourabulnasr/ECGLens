"""Model workers, each run as its own subprocess (`python -m ecglens.text_worker <cmd> ...`).

Loading MarianMT and MiniLM in one interpreter has crashed this sandbox before (see README
"Engineering notes"), so the pipeline always calls these through `run_worker()`.

  translate IN.json OUT.json   list[str] German -> {german: english}   (opus-mt-de-en, greedy)
  embed     IN.json OUT.npy    list[str] -> float32 [n, 384] L2-normalised (all-MiniLM-L6-v2)
"""
import json
import subprocess
import sys

import numpy as np

MARIAN = "Helsinki-NLP/opus-mt-de-en"
MINILM = "sentence-transformers/all-MiniLM-L6-v2"


def run_worker(*args: str) -> None:
    subprocess.run([sys.executable, "-m", "ecglens.text_worker", *args], check=True)


def _translate(src: str, dst: str, batch: int = 32) -> None:
    import torch
    from transformers import MarianMTModel, MarianTokenizer

    torch.set_num_threads(6)
    texts = json.load(open(src, encoding="utf-8"))
    tok = MarianTokenizer.from_pretrained(MARIAN)
    model = MarianMTModel.from_pretrained(MARIAN).eval()
    # sort by length so batches pad little
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = {}
    try:
        out = json.load(open(dst, encoding="utf-8"))  # resume support
    except FileNotFoundError:
        pass
    todo = [i for i in order if texts[i] not in out]
    for b in range(0, len(todo), batch):
        idx = todo[b : b + batch]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=192)
        with torch.no_grad():
            gen = model.generate(**enc, num_beams=1, max_new_tokens=160)
        for i, t in zip(idx, tok.batch_decode(gen, skip_special_tokens=True)):
            out[texts[i]] = t
        if (b // batch) % 20 == 0:
            json.dump(out, open(dst, "w", encoding="utf-8"), ensure_ascii=False)
            print(f"translated {len(out)}/{len(texts)}", flush=True)
    json.dump(out, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print(f"translated {len(out)}/{len(texts)} done", flush=True)


def _embed(src: str, dst: str) -> None:
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(6)
    texts = json.load(open(src, encoding="utf-8"))
    model = SentenceTransformer(MINILM, device="cpu")
    emb = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    np.save(dst, emb.astype(np.float32))
    print(f"embedded {len(texts)} -> {emb.shape}", flush=True)


if __name__ == "__main__":
    cmd, src, dst = sys.argv[1:4]
    {"translate": _translate, "embed": _embed}[cmd](src, dst)

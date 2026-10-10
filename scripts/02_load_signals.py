"""Extract records100/ from the PTB-XL zip and pack the 100 Hz waveforms into one int16 array.

int16 is lossless: PTB-XL is stored with 1000 ADC units per mV, and we standardise per lead at load time.
Output: data/signals_100.npy [N, 12, 1000] int16, data/signals_ids.npy (ecg_id per row, sorted).
"""
import os
import sys
import zipfile

import numpy as np
import wfdb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ecglens.data import build_tables

ZIP = "data/raw/ptb-xl-data.zip"
df, _ = build_tables(with_text=False)
ids = np.array(sorted(df.index))

root = "data/raw/extracted"
if not os.path.isdir(os.path.join(root, "records100")):
    with zipfile.ZipFile(ZIP) as z:
        names = z.namelist()
        print("zip entries:", len(names), names[:3])
        members = [n for n in names if "records100/" in n and not n.endswith("/")]
        print("records100 members:", len(members))
        z.extractall(root, members)
# locate records100 (zip may wrap it in a top-level folder)
base = None
for dp, dn, _ in os.walk(root):
    if "records100" in dn:
        base = dp
        break
print("base:", base)

X = np.zeros((len(ids), 12, 1000), dtype=np.int16)
for k, ecg_id in enumerate(ids):
    rec = wfdb.rdrecord(os.path.join(base, df.loc[ecg_id, "filename_lr"]), physical=False)
    X[k] = rec.d_signal.T
    if k % 4000 == 0:
        print(k, X[k].std(), flush=True)
np.save("data/signals_100.npy", X)
np.save("data/signals_ids.npy", ids)
print("saved", X.shape, X.dtype, "gain check:", rec.adc_gain[:3])

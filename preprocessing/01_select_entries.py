"""Step 1: choose PDB entries from the wwPDB index (method, resolution, date window).

Writes data/entries.tsv  (pdb_id, date, resolution, method, era) where era in {pre, post}:
  pre  = deposited before cfg.pre_cutoff        -> train/val/test (random split by sequence cluster)
  post = [pre_cutoff, post_end]                 -> temporal hold-out test (MPBind 'Test2')
"""
import argparse
import os

import pandas as pd

from ppi_prep.config import CFG
from ppi_prep.net import WWPDB, fetch

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--index", default=None, help="local entries.idx (skip download)")
a = ap.parse_args()

idx = a.index or os.path.join(a.data, "raw", "entries.idx")
if not a.index and not fetch(f"{WWPDB}/derived_data/index/entries.idx", idx):
    raise SystemExit("could not download entries.idx")

rows = []
with open(idx, encoding="latin-1") as fh:
    next(fh), next(fh)                       # header + dashes
    for line in fh:
        f = line.rstrip("\n").split("\t")
        if len(f) < 8:
            continue
        pid, date, res, method = f[0].strip(), f[2].strip(), f[6].strip(), f[7].strip()
        try:
            res = float(res.split(",")[0])
        except ValueError:
            continue                          # NMR / no resolution
        rows.append((pid.lower(), pd.to_datetime(date, format="%m/%d/%y", errors="coerce"), res, method))

df = pd.DataFrame(rows, columns=["pdb_id", "date", "resolution", "method"]).dropna()
# %y pivots 69->1969/68->2068; PDB dates are all <= 2026 so fix any future date
df.loc[df.date > pd.Timestamp.today(), "date"] -= pd.DateOffset(years=100)
df = df[(df.resolution <= CFG.max_resolution) & df.method.str.upper().isin(CFG.methods)]
cut, end = pd.Timestamp(CFG.pre_cutoff), pd.Timestamp(CFG.post_end)
df["era"] = None
df.loc[df.date < cut, "era"] = "pre"
df.loc[(df.date >= cut) & (df.date <= end), "era"] = "post"
df = df.dropna(subset=["era"]).sort_values("pdb_id")
os.makedirs(a.data, exist_ok=True)
df.to_csv(os.path.join(a.data, "entries.tsv"), sep="\t", index=False, date_format="%Y-%m-%d")
print(df.era.value_counts().to_string())

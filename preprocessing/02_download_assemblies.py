"""Step 2: download biological assembly 1 (mmCIF) for every selected entry (parallel, resumable)."""
import argparse
import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from tqdm import tqdm

from ppi_prep.config import CFG
from ppi_prep.net import assembly_url, fetch

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--limit", type=int, default=0, help="debug: only first N entries")
a = ap.parse_args()

ids = pd.read_csv(os.path.join(a.data, "entries.tsv"), sep="\t").pdb_id.tolist()
if a.limit:
    ids = ids[:a.limit]


def job(pid):
    ok = fetch(assembly_url(pid, CFG.assembly_id), os.path.join(a.data, "raw", "assemblies", f"{pid}.cif.gz"))
    return pid, ok


with ThreadPoolExecutor(a.workers) as ex:
    res = list(tqdm(ex.map(job, ids), total=len(ids)))
missing = [p for p, ok in res if not ok]
print(f"downloaded/cached {len(res) - len(missing)}, missing {len(missing)}")
open(os.path.join(a.data, "missing_assemblies.txt"), "w").write("\n".join(missing))

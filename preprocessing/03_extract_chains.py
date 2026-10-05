"""Step 3: per assembly -> clean protein chains + heavy-atom 5 A interface labels -> data/chains/*.npz
and data/chains.tsv (one row per kept chain). Works on downloaded assemblies or any local pdb/cif files."""
import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from tqdm import tqdm

from ppi_prep.config import CFG
from ppi_prep.structure import break_fraction, extract_chains, label_interfaces, load_structure


def process(args):
    path, out_dir = args
    pid = os.path.basename(path).split(".")[0].split("-")[0].lower()
    try:
        chains = extract_chains(load_structure(path), min_res=CFG.min_partner_res)
        if sum(len(c.atoms) for c in chains) > CFG.max_assembly_atoms:
            return pid, [], "too_large"
        label_interfaces(chains, CFG.contact_cutoff)
    except Exception as e:                                   # corrupt/odd file: log, never crash the run
        return pid, [], f"error:{type(e).__name__}"
    rows = []
    for ch in chains:
        n_if = int(ch.label.sum())
        if not (CFG.min_len <= len(ch) <= CFG.max_len):
            continue
        if n_if < CFG.min_interface_res or break_fraction(ch) > CFG.max_break_frac:
            continue
        uid = f"{pid}_{ch.chain_id}"
        np.savez_compressed(os.path.join(out_dir, uid + ".npz"), seq=ch.seq, resseq=ch.resseq,
                            icode=np.array(ch.icode), bb=ch.bb, cb=ch.cb, sc=ch.sc, label=ch.label)
        rows.append(dict(uid=uid, pdb_id=pid, chain=ch.chain_id, length=len(ch), n_interface=n_if,
                         frac_interface=n_if / len(ch), n_partners=len(ch.partners), seq=ch.seq))
    return pid, rows, "ok"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--local", default=None, help="dir of local .pdb/.cif(.gz) files instead of downloaded assemblies")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()

    src = a.local or os.path.join(a.data, "raw", "assemblies")
    files = sorted(f for f in glob.glob(os.path.join(src, "*")) if f.lower().endswith((".pdb", ".cif", ".gz", ".mmcif")))
    out_dir = os.path.join(a.data, "chains")
    os.makedirs(out_dir, exist_ok=True)
    rows, status = [], {}
    with ProcessPoolExecutor(a.workers) as ex:
        for pid, r, st in tqdm(ex.map(process, [(f, out_dir) for f in files], chunksize=8), total=len(files)):
            rows += r
            status[st] = status.get(st, 0) + 1
    df = pd.DataFrame(rows)
    if not a.local and os.path.exists(os.path.join(a.data, "entries.tsv")):
        df = df.merge(pd.read_csv(os.path.join(a.data, "entries.tsv"), sep="\t"), on="pdb_id", how="left")
    df.to_csv(os.path.join(a.data, "chains.tsv"), sep="\t", index=False)
    print(f"{len(df)} chains from {df.pdb_id.nunique() if len(df) else 0} entries; status={status}")

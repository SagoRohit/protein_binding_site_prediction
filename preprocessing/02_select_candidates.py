"""Step 2a (cluster-first): decide WHICH entries to download.

Cluster every protein SEQRES sequence of the selected entries at 30 % identity (identical sequences collapsed first),
then pick the smallest set of entries that covers each cluster with `cover_per_cluster` good entries (greedy set cover).
Post-era chains that hit any pre-era sequence at >=30 % are ignored (they could never enter the temporal test set).
Output: data/selected_entries.tsv  (pdb_id, era, resolution)
Final redundancy removal / splitting is still done on the *modelled* chains in step 4.
"""
import argparse
import os

import pandas as pd

from ppi_prep import mmseqs
from ppi_prep.config import CFG
from ppi_prep.net import WWPDB, fetch
from ppi_prep.select import greedy_cover, read_seqres

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--seqres", default=None, help="local pdb_seqres.txt(.gz) (skip download)")
a = ap.parse_args()

entries = pd.read_csv(os.path.join(a.data, "entries.tsv"), sep="\t")
path = a.seqres or os.path.join(a.data, "raw", "pdb_seqres.txt.gz")
if not a.seqres and not fetch(f"{WWPDB}/derived_data/pdb_seqres.txt.gz", path):
    raise SystemExit("could not download pdb_seqres.txt.gz")

sq = read_seqres(path)
sq["n_chains"] = sq.groupby("pdb_id").chain.transform("size")          # protein chains per entry (asym unit)
sq = sq.merge(entries[["pdb_id", "era", "resolution"]], on="pdb_id")   # only selected entries
sq["length"] = sq.seq.str.len()
sq = sq[(sq.length >= CFG.min_len) & (sq.length <= CFG.max_len * CFG.seqres_len_slack)]
print(f"{len(sq)} candidate chains in {sq.pdb_id.nunique()} entries")

# unique sequences -> ids
uniq = sq.seq.drop_duplicates().reset_index(drop=True)
sid = pd.Series(uniq.index, index=uniq.values)
sq["sid"] = sq.seq.map(sid)
u = pd.DataFrame({"uid": [f"s{i}" for i in uniq.index], "seq": uniq.values})
pre_seqs = set(sq[sq.era == "pre"].seq)
chosen = set()

for era in ("pre", "post"):
    e = sq[sq.era == era]
    if era == "post":
        post_u = u[u.seq.isin(set(e.seq)) & ~u.seq.isin(pre_seqs)]     # exact pre duplicates are leaks
        leaked = mmseqs.hits(post_u, u[u.seq.isin(pre_seqs)], CFG.seq_id, CFG.leak_coverage, a.threads)
        keep = set(post_u[~post_u.uid.isin(leaked)].seq)
        print(f"post-era: {len(leaked)} unique sequences similar to pre-era dropped")
        e = e[e.seq.isin(keep)]
    if not len(e):
        continue
    sub = u[u.seq.isin(set(e.seq))]
    cl = mmseqs.cluster(sub, CFG.seq_id, CFG.coverage, a.threads)
    e = e.assign(cluster=("s" + e.sid.astype(str)).map(cl))
    sel = greedy_cover(e[["cluster", "pdb_id", "resolution", "n_chains"]], CFG.cover_per_cluster)
    print(f"{era}: {e.cluster.nunique()} clusters -> {len(sel)} entries to download")
    chosen |= {(p, era) for p in sel}

out = pd.DataFrame(sorted(chosen), columns=["pdb_id", "era"]).merge(entries[["pdb_id", "resolution"]], on="pdb_id")
out.to_csv(os.path.join(a.data, "selected_entries.tsv"), sep="\t", index=False)
print(f"selected {len(out)} entries")

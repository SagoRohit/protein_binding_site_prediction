"""Cluster-first entry selection: pick as few PDB entries as possible such that every 30%-identity
sequence cluster is represented by `k` good entries (greedy set cover). Pure functions, no I/O."""
import gzip
from collections import defaultdict

import pandas as pd


def read_seqres(path):
    """pdb_seqres.txt(.gz) -> DataFrame(pdb_id, chain, seq) for mol:protein chains only."""
    opener = gzip.open if path.endswith(".gz") else open
    rows, hdr = [], None
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                f = line[1:].split()
                hdr = tuple(f[0].split("_", 1)) if "mol:protein" in f[1:2] else None
            elif hdr:
                rows.append((hdr[0].lower(), hdr[1], line.strip()))
                hdr = None
    return pd.DataFrame(rows, columns=["pdb_id", "chain", "seq"])


def greedy_cover(cands, k):
    """cands: DataFrame(cluster, pdb_id, resolution, n_chains), one row per (cluster, entry) pair.
    Returns the set of selected pdb_ids such that each cluster has min(k, #available) selected entries.
    Entry preference: has >=2 protein chains (can have an interface) > better resolution > more chains."""
    cands = cands.drop_duplicates(["cluster", "pdb_id"])
    best = cands.assign(multi=(cands.n_chains >= 2)).sort_values(
        ["multi", "resolution", "n_chains", "pdb_id"], ascending=[False, True, False, True])
    per_cluster = {c: g.pdb_id.tolist() for c, g in best.groupby("cluster", sort=False)}
    entry_clusters = defaultdict(set)
    for c, p in zip(cands.cluster, cands.pdb_id):
        entry_clusters[p].add(c)
    selected, count = set(), defaultdict(int)
    for c in sorted(per_cluster, key=lambda c: (len(per_cluster[c]), str(c))):   # scarcest clusters first
        for p in per_cluster[c]:
            if count[c] >= k:
                break
            if p in selected:
                continue
            selected.add(p)
            for c2 in entry_clusters[p]:
                count[c2] += 1
    return selected

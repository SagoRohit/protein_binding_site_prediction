"""Step 5: sanity report on the final dataset (sizes, lengths, positive rate, split leakage by entry)."""
import argparse
import os

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
a = ap.parse_args()
ch = pd.read_csv(os.path.join(a.data, "chains.tsv"), sep="\t")
sp = pd.read_csv(os.path.join(a.data, "splits.tsv"), sep="\t").merge(ch, on="uid")
g = sp.groupby("split")
rep = pd.DataFrame({
    "chains": g.size(), "entries": g.pdb_id.nunique(),
    "len_median": g.length.median(), "len_max": g.length.max(),
    "interface_frac_mean": g.frac_interface.mean(),
    "residues": g.length.sum(),
    "pos_rate": g.n_interface.sum() / g.length.sum(),
})
print(rep.round(3).to_string())
for s1 in rep.index:
    for s2 in rep.index:
        if s1 < s2:
            shared = set(sp[sp.split == s1].pdb_id) & set(sp[sp.split == s2].pdb_id)
            if shared:
                print(f"note: {len(shared)} PDB entries contribute chains to both {s1} and {s2} "
                      f"(different, <30%-identity chains of the same complex)")
rep.to_csv(os.path.join(a.data, "stats.tsv"), sep="\t")

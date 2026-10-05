"""Step 4: redundancy removal + leakage-free splits.

pre-era chains  -> MMseqs2 30% clusters -> keep `members_per_cluster` per cluster (best resolution)
                -> split WHOLE clusters 70/15/15 into train/val/test
                -> any val/test chain still hitting train at >=30% id is removed (safety net)
post-era chains -> remove anything with a >=30% hit to ANY pre-era chain, cluster at 30%, 1 per cluster
                -> test_temporal (MPBind 'Test2' analogue)
Output: data/splits.tsv (uid, split, cluster) and data/{split}.txt uid lists.
"""
import argparse
import os

import numpy as np
import pandas as pd

from ppi_prep import mmseqs
from ppi_prep.config import CFG

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--threads", type=int, default=8)
a = ap.parse_args()

df = pd.read_csv(os.path.join(a.data, "chains.tsv"), sep="\t")
rng = np.random.default_rng(CFG.seed)


def pick(sub, cl, k):
    sub = sub.assign(cluster=sub.uid.map(cl))
    sub = sub.sort_values(["cluster", "resolution", "n_interface", "uid"], ascending=[True, True, False, True])
    return sub.groupby("cluster").head(k)


# ---------------- pre-era: train / val / test ----------------
pre = df[df.era == "pre"]
cl = mmseqs.cluster(pre, CFG.seq_id, CFG.coverage, a.threads)
pre = pick(pre, cl, CFG.members_per_cluster).copy()
clusters = np.array(sorted(pre.cluster.unique()))
rng.shuffle(clusters)
b = np.cumsum(CFG.split_frac) * len(clusters)
assign = {c: ("train" if i < b[0] else "val" if i < b[1] else "test") for i, c in enumerate(clusters)}
pre["split"] = pre.cluster.map(assign)

train = pre[pre.split == "train"]
for s in ("val", "test"):                                     # greedy clustering is approximate -> verify
    sub = pre[pre.split == s]
    leaked = mmseqs.hits(sub, train, CFG.seq_id, CFG.leak_coverage, a.threads)
    print(f"{s}: removed {len(leaked)} chains similar to train")
    pre = pre[~pre.uid.isin(leaked)]
leaked = mmseqs.hits(pre[pre.split == "test"], pre[pre.split == "val"], CFG.seq_id, CFG.leak_coverage, a.threads)
pre = pre[~pre.uid.isin(leaked)]

# ---------------- post-era: temporal test ----------------
post = df[df.era == "post"]
leaked = mmseqs.hits(post, df[df.era == "pre"], CFG.seq_id, CFG.leak_coverage, a.threads)
post = post[~post.uid.isin(leaked)]
print(f"post-era: {len(leaked)} chains removed for similarity to pre-era; {len(post)} left")
if len(post):
    post = pick(post, mmseqs.cluster(post, CFG.seq_id, CFG.coverage, a.threads), 1)
    post["split"] = "test_temporal"

out = pd.concat([pre, post])[["uid", "split", "cluster"]]
out.to_csv(os.path.join(a.data, "splits.tsv"), sep="\t", index=False)
for s, g in out.groupby("split"):
    open(os.path.join(a.data, f"{s}.txt"), "w").write("\n".join(g.uid))
print(out.split.value_counts().to_string())

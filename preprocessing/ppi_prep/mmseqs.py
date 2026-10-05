import os
import shutil
import subprocess
import tempfile


def _run(cmd):
    if shutil.which("mmseqs") is None:
        raise SystemExit("mmseqs not found. Install: conda install -c bioconda mmseqs2")
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def write_fasta(df, path):
    with open(path, "w") as fh:
        for uid, seq in zip(df.uid, df.seq):
            fh.write(f">{uid}\n{seq}\n")


def cluster(df, seq_id, cov, threads=8):
    """MMseqs2 greedy clustering. Returns {member_uid: representative_uid}."""
    with tempfile.TemporaryDirectory() as tmp:
        fa = os.path.join(tmp, "in.fa")
        write_fasta(df, fa)
        _run(["mmseqs", "easy-cluster", fa, os.path.join(tmp, "out"), os.path.join(tmp, "t"),
              "--min-seq-id", str(seq_id), "-c", str(cov), "--cov-mode", "0", "--threads", str(threads)])
        m = {}
        for line in open(os.path.join(tmp, "out_cluster.tsv")):
            rep, mem = line.split()
            m[mem] = rep
    return m


def hits(query_df, target_df, seq_id, cov, threads=8):
    """uids of query chains having ANY alignment to target with identity >= seq_id (sensitive search)."""
    if len(query_df) == 0 or len(target_df) == 0:
        return set()
    with tempfile.TemporaryDirectory() as tmp:
        q, t, out = (os.path.join(tmp, x) for x in ("q.fa", "t.fa", "o.m8"))
        write_fasta(query_df, q)
        write_fasta(target_df, t)
        _run(["mmseqs", "easy-search", q, t, out, os.path.join(tmp, "t"), "--min-seq-id", str(seq_id),
              "-c", str(cov), "--cov-mode", "0", "-s", "7.5", "--max-seqs", "1000",
              "--threads", str(threads), "--format-output", "query,target,pident"])
        return {l.split("\t")[0] for l in open(out)}

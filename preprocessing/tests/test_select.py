import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ppi_prep.select import greedy_cover, read_seqres


def rows(*r):
    return pd.DataFrame(r, columns=["cluster", "pdb_id", "resolution", "n_chains"])


def test_cover_prefers_shared_entries_and_multichain():
    # entry e1 covers clusters 1 and 2 -> one download covers both with k=1
    c = rows((1, "e1", 2.0, 2), (2, "e1", 2.0, 2), (1, "e2", 1.0, 1), (2, "e3", 1.0, 1))
    assert greedy_cover(c, 1) == {"e1"}


def test_cover_k_and_fallback():
    c = rows((1, "a", 1.0, 2), (1, "b", 2.0, 2), (1, "c", 3.0, 2))
    assert greedy_cover(c, 2) == {"a", "b"}                 # best resolution first
    assert greedy_cover(rows((1, "a", 1.0, 2)), 3) == {"a"}  # fewer than k available


def test_read_seqres(tmp_path):
    f = tmp_path / "s.txt"
    f.write_text(">101m_A mol:protein length:4  MYO\nACDE\n>1abc_B mol:na length:3  DNA\nACG\n>1abc_A mol:protein length:2  X\nGG\n")
    df = read_seqres(str(f))
    assert df.values.tolist() == [["101m", "A", "ACDE"], ["1abc", "A", "GG"]]

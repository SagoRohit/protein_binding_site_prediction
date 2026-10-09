import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ppi_prep.structure import Chain, extract_chains, label_interfaces, load_structure, virtual_cb

REPO = os.path.join(os.path.dirname(__file__), "..", "..")


def toy_chain(cid, xs):
    """One heavy atom per residue at x = xs[i] (y = z = 0)."""
    atoms = np.array([[x, 0, 0] for x in xs], dtype=np.float32)
    n = len(xs)
    bb = np.zeros((n, 4, 3), dtype=np.float32)
    return Chain(cid, "A" * n, np.arange(n), [""] * n, bb, atoms, atoms, atoms, np.arange(n, dtype=np.int32))


def test_cutoff_is_heavy_atom_based():
    a, b = toy_chain("A", [0, 20, 40]), toy_chain("B", [4.9, 100, 200])
    label_interfaces([a, b], cutoff=5.0)
    assert a.label.tolist() == [1, 0, 0] and b.label.tolist() == [1, 0, 0]
    assert a.partners == {"B"}


def test_just_outside_cutoff():
    a, b = toy_chain("A", [0, 20, 40]), toy_chain("B", [5.1, 100, 200])
    label_interfaces([a, b], cutoff=5.0)
    assert a.label.sum() == 0 and b.label.sum() == 0


def test_same_chain_contacts_ignored_and_single_chain():
    a = toy_chain("A", [0, 1, 2])                      # intra-chain clash must not label anything
    label_interfaces([a], cutoff=5.0)
    assert a.label.sum() == 0


def _brs():
    return load_structure(os.path.join(REPO, "code", "1BRS.pdb"))


@pytest.mark.skipif(not os.path.exists(os.path.join(REPO, "code", "1BRS.pdb")), reason="1BRS.pdb missing")
def test_virtual_cb_matches_real_cb():
    from Bio.PDB.Polypeptide import is_aa
    errs = []
    for res in _brs()[0]["A"]:
        if is_aa(res, standard=True) and res.resname != "GLY" and all(k in res for k in ("N", "CA", "C", "CB")):
            v = virtual_cb(res["N"].coord.astype(float), res["CA"].coord.astype(float), res["C"].coord.astype(float))
            errs.append(np.linalg.norm(v - res["CB"].coord))
    assert np.mean(errs) < 0.15


@pytest.mark.skipif(not os.path.exists(os.path.join(REPO, "code", "1BRS.pdb")), reason="1BRS.pdb missing")
def test_1brs_barnase_barstar():
    s = _brs()
    chains = {c.chain_id: c for c in extract_chains(s, min_res=5)}
    assert set(chains) == {"A", "B", "C", "D", "E", "F"}
    label_interfaces(list(chains.values()), 5.0)
    # barnase (A,B,C) pairs with barstar (D,E,F); A<->D is the famous interface (~20 residues per side).
    # NB: this file is the asymmetric unit, so A also touches B/C through crystal contacts -> why we use
    # biological assemblies in the real pipeline.
    assert chains["D"].partners == {"A"} and chains["E"].partners == {"B"}
    for k in "DEF":                                    # barstar: ~20 interface residues, no crystal contacts
        assert 15 <= chains[k].label.sum() <= 25
    assert chains["A"].label.sum() > chains["D"].label.sum()   # barnase also has crystal contacts here

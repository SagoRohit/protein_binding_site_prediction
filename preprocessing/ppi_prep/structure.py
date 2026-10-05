"""Structure parsing + protein-protein interface labelling (heavy-atom, 5 A, other protein chains)."""
import gzip
import os
from dataclasses import dataclass, field

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser
from scipy.spatial import cKDTree

AA3 = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
       "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
AA1 = "ARNDCQEGHILKMFPSTWYV"
AA3_TO_1 = dict(zip(AA3, AA1))
MODRES = {"MSE": "MET"}          # selenomethionine -> Met
BACKBONE = ("N", "CA", "C", "O")


@dataclass
class Chain:
    chain_id: str
    seq: str
    resseq: np.ndarray            # (L,) author residue numbers
    icode: list
    bb: np.ndarray                # (L, 4, 3)  N, CA, C, O
    cb: np.ndarray                # (L, 3)     real CB, virtual CB for Gly / missing
    sc: np.ndarray                # (L, 3)     side-chain heavy-atom centroid (CA for Gly)
    atoms: np.ndarray             # (A, 3)     all heavy atoms
    atom_res: np.ndarray          # (A,)       residue index of each atom
    label: np.ndarray = field(default=None)   # (L,) int8 interface label
    partners: set = field(default_factory=set)

    def __len__(self):
        return len(self.seq)


def load_structure(path):
    opener = gzip.open if path.endswith(".gz") else open
    base = path[:-3] if path.endswith(".gz") else path
    sid = os.path.basename(base).split(".")[0].split("-")[0]
    with opener(path, "rt") as fh:
        if base.lower().endswith((".cif", ".mmcif")):
            # label_asym_id is unique per chain instance inside assembly files (auth ids can collide)
            return MMCIFParser(QUIET=True, auth_chains=False).get_structure(sid, fh)
        return PDBParser(QUIET=True).get_structure(sid, fh)


def virtual_cb(n, ca, c):
    b, cc = ca - n, c - ca
    a = np.cross(b, cc)
    return -0.58273431 * a + 0.56802827 * b - 0.54067466 * cc + ca


def _residue_ok(res):
    name = res.resname.strip()
    return (name in AA3_TO_1 or name in MODRES) and all(a in res for a in BACKBONE)


def extract_chains(structure, min_res=1, model_idx=0):
    """All protein chains of one model with >= min_res usable residues."""
    out = []
    for chain in structure[model_idx]:
        seq, resseq, icode, bb, cb, sc, atoms, atom_res = [], [], [], [], [], [], [], []
        for res in chain:
            if res.id[0] == "W" or not _residue_ok(res):
                continue
            name = MODRES.get(res.resname.strip(), res.resname.strip())
            heavy = [a for a in res if (a.element or "").strip() not in ("H", "D")]
            idx = len(seq)
            n, ca, c, o = (res[k].coord.astype(np.float64) for k in BACKBONE)
            side = [a.coord for a in heavy if a.get_id() not in BACKBONE]
            seq.append(AA3_TO_1[name])
            resseq.append(res.id[1])
            icode.append(res.id[2].strip())
            bb.append(np.stack([n, ca, c, o]))
            cb.append(res["CB"].coord if "CB" in res else virtual_cb(n, ca, c))
            sc.append(np.mean(side, axis=0) if side else ca)
            atoms.extend(a.coord for a in heavy)
            atom_res.extend([idx] * len(heavy))
        if len(seq) >= min_res:
            out.append(Chain(str(chain.id), "".join(seq), np.array(resseq), icode,
                             np.array(bb, dtype=np.float32), np.array(cb, dtype=np.float32),
                             np.array(sc, dtype=np.float32), np.array(atoms, dtype=np.float32),
                             np.array(atom_res, dtype=np.int32)))
    return out


def label_interfaces(chains, cutoff=5.0):
    """Residue = interface (1) if any heavy atom is within `cutoff` A of a heavy atom of ANOTHER chain
    in `chains` (callers pass only protein chains of one biological assembly)."""
    for ch in chains:
        ch.label = np.zeros(len(ch), dtype=np.int8)
        ch.partners = set()
    if len(chains) < 2:
        return chains
    xyz = np.concatenate([c.atoms for c in chains])
    cid = np.concatenate([np.full(len(c.atoms), i, dtype=np.int32) for i, c in enumerate(chains)])
    rid = np.concatenate([c.atom_res for c in chains])
    pairs = cKDTree(xyz).query_pairs(r=cutoff, output_type="ndarray")
    if len(pairs):
        pairs = pairs[cid[pairs[:, 0]] != cid[pairs[:, 1]]]
    for i, j in pairs:
        ci, cj = chains[cid[i]], chains[cid[j]]
        ci.label[rid[i]] = 1
        cj.label[rid[j]] = 1
        ci.partners.add(cj.chain_id)
        cj.partners.add(ci.chain_id)
    return chains


def break_fraction(ch, thresh=4.5):
    """Fraction of sequence-consecutive residues whose CA-CA distance exceeds `thresh` (unmodelled gaps)."""
    if len(ch) < 2:
        return 0.0
    d = np.linalg.norm(np.diff(ch.bb[:, 1], axis=0), axis=1)
    return float((d > thresh).mean())

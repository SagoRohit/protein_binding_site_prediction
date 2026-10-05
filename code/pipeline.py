"""
Minimal, from-scratch PPI structure pipeline.
Stages:
  1. parse a PDB/mmCIF file with BioPython
  2. pull out per-chain residue lists (CA coordinate + amino-acid identity)
  3. compute an NA x NB distance matrix between two chains
  4. threshold it into a 0/1 contact matrix
  5. collapse that into per-residue 0/1 interface labels
  6. build a PyTorch Geometric graph per chain (nodes=residues, edges=structural contacts, y=interface label)
"""

import numpy as np
from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB.Polypeptide import is_aa

# 20 standard amino acids -> index, for one-hot node features
AA3 = ["ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
       "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"]
AA_INDEX = {aa: i for i, aa in enumerate(AA3)}


def parse_structure(path, structure_id="X"):
    """Step 1: load a .pdb or .cif file into a Bio.PDB Structure object."""
    if path.lower().endswith(".cif"):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    return parser.get_structure(structure_id, path)


def get_chain_residues(structure, chain_id, model_idx=0):
    """
    Step 2: pull real amino-acid residues (skip waters/ligands/hetero groups)
    out of one chain. Returns a list of dicts, one per residue, each holding
    the residue's CA coordinate and its amino-acid one-hot index.
    Residues without a CA atom (rare, e.g. disordered) are skipped.
    """
    chain = structure[model_idx][chain_id]
    residues = []
    for res in chain:
        if not is_aa(res, standard=True):
            continue  # drops HOH, ligands, ions, non-standard residues
        if "CA" not in res:
            continue
        residues.append({
            "resseq": res.id[1],
            "resname": res.resname,
            "aa_index": AA_INDEX.get(res.resname, 20),  # 20 = "unknown"
            "coord": res["CA"].coord,  # numpy array, shape (3,)
        })
    return residues


def compute_distance_matrix(residues_a, residues_b):
    """
    Step 3: the actual N_A x N_B distance matrix, done ourselves with numpy
    broadcasting (no hidden library magic) using CA-CA distance.
    """
    coords_a = np.array([r["coord"] for r in residues_a])  # (NA, 3)
    coords_b = np.array([r["coord"] for r in residues_b])  # (NB, 3)
    # (NA,1,3) - (1,NB,3) -> (NA,NB,3) -> norm over last axis -> (NA,NB)
    diff = coords_a[:, None, :] - coords_b[None, :, :]
    dist_matrix = np.linalg.norm(diff, axis=-1)
    return dist_matrix


def compute_contact_matrix(dist_matrix, threshold=8.0):
    """Step 4: threshold distances into a 0/1 contact matrix."""
    return (dist_matrix <= threshold).astype(int)


def compute_interface_labels(contact_matrix):
    """
    Step 5: a residue is an "interface residue" if it has >=1 contact
    to ANY residue on the other chain. Row-OR for chain A, column-OR for chain B.
    """
    labels_a = (contact_matrix.sum(axis=1) > 0).astype(int)  # (NA,)
    labels_b = (contact_matrix.sum(axis=0) > 0).astype(int)  # (NB,)
    return labels_a, labels_b


def build_intra_chain_edges(residues, threshold=8.0):
    """
    Structural graph WITHIN one chain: two residues get an edge if their
    CAs are within `threshold` Angstroms of each other (this is the graph
    a GNN would actually see at inference time -- it only needs ONE chain's
    own structure, not the partner's).
    Returns edge_index in PyG's [2, num_edges] COO format (both directions).
    """
    n = len(residues)
    coords = np.array([r["coord"] for r in residues])
    diff = coords[:, None, :] - coords[None, :, :]
    d = np.linalg.norm(diff, axis=-1)
    contact = (d <= threshold) & (d > 0)  # exclude self-loops
    src, dst = np.nonzero(contact)
    edge_index = np.stack([src, dst], axis=0)  # shape (2, num_edges)
    return edge_index


def one_hot_features(residues):
    """Node feature matrix: (N, 21) one-hot amino-acid identity."""
    x = np.zeros((len(residues), 21), dtype=np.float32)
    for i, r in enumerate(residues):
        x[i, r["aa_index"]] = 1.0
    return x


def build_pyg_data(residues, labels, edge_index):
    """
    Step 6: package one chain into a torch_geometric.data.Data object.
    x        : (N, 21)  node features (one-hot amino acid)
    edge_index: (2, E)  intra-chain structural contacts
    y        : (N,)     interface label (0/1), derived from the COMPLEX
    """
    import torch
    from torch_geometric.data import Data
    x = torch.tensor(one_hot_features(residues), dtype=torch.float)
    ei = torch.tensor(edge_index, dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.long)
    return Data(x=x, edge_index=ei, y=y)


if __name__ == "__main__":
    structure = parse_structure("toy_complex.pdb", structure_id="TOY")
    res_a = get_chain_residues(structure, "A")
    res_b = get_chain_residues(structure, "B")

    print(f"Chain A: {len(res_a)} residues -> {[r['resname'] for r in res_a]}")
    print(f"Chain B: {len(res_b)} residues -> {[r['resname'] for r in res_b]}")

    dist = compute_distance_matrix(res_a, res_b)
    print("\nDistance matrix (A x B), Angstroms:")
    print(np.round(dist, 2))

    contact = compute_contact_matrix(dist, threshold=8.0)
    print("\nContact matrix (threshold=8.0 A):")
    print(contact)

    labels_a, labels_b = compute_interface_labels(contact)
    print("\nInterface labels A:", dict(zip([r["resname"] for r in res_a], labels_a)))
    print("Interface labels B:", dict(zip([r["resname"] for r in res_b], labels_b)))

    edges_a = build_intra_chain_edges(res_a, threshold=8.0)
    print("\nChain A intra-chain edge_index (backbone contacts):")
    print(edges_a)

"""Single place for every dataset-defining constant (all follow PeSTo / MPBind unless noted)."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # ---- entry selection (PDB-level) ----
    max_resolution: float = 3.0          # A; X-ray/EM only. PeSTo/MPBind apply no filter -> our choice, see context.md
    methods: tuple = ("X-RAY DIFFRACTION", "ELECTRON MICROSCOPY")
    pre_cutoff: str = "2022-01-01"       # MPBind Dataset1 cutoff -> train/val/test(random cluster split) are BEFORE this
    post_end: str = "2024-06-21"         # MPBind Dataset2 end date -> temporal test set is [pre_cutoff, post_end]
    assembly_id: int = 1                 # first biological assembly only (crystal contacts are not interfaces)

    # ---- chain selection ----
    min_len: int = 48                    # MPBind min residues
    max_len: int = 1000                  # ~MPBind 8192-atom cap; also fits ESM-style 1022 limit
    min_partner_res: int = 5             # a partner protein chain must have >= this many residues (drops tiny fragments)
    min_interface_res: int = 5           # keep chain only if >= this many interface residues (set 0 to keep non-binders)
    max_break_frac: float = 0.10         # drop chain if >10% consecutive CA-CA gaps are > 4.5 A (unmodelled loops)
    max_assembly_atoms: int = 250_000    # skip giant assemblies (capsids etc.)

    # ---- labels ----
    contact_cutoff: float = 5.0          # A, min heavy-atom distance between residues of different protein chains

    # ---- clustering / splitting ----
    seq_id: float = 0.30                 # MMseqs2 --min-seq-id
    coverage: float = 0.80               # MMseqs2 -c (cov-mode 0, both ways)
    leak_coverage: float = 0.5           # stricter coverage when searching held-out chains against train (any hit >= seq_id is removed)
    split_frac: tuple = (0.70, 0.15, 0.15)   # train/val/test by CLUSTER (PeSTo ratios)
    members_per_cluster: int = 1         # 1 = non-redundant at 30% (moderate size). Raise to enlarge dataset.
    seed: int = 0


CFG = Config()

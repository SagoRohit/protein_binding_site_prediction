# Preprocessing — context, goal, next steps

## Project goal
Build a **novel architecture for protein–protein binding-site (interface) prediction** (residue-level, single chain in → per-residue
interface probability out). The rough architecture idea lives in `literature review/Confidence_Calibrated_Biophysical_GNN_Research_Proposal`
(PLM embeddings + biophysical priors + confidence-gated geometric GNN). We will **not** copy that proposal 1:1 (it targets protein–*ligand*
sites); we start with PPI sites and treat the architecture as a hypothesis to be tested step by step.

## Why preprocessing first
Whatever architecture we end up with, the *dataset* (which chains, how labels are defined, how splits are made) should follow the standard
literature protocol (PeSTo, MPBind, ScanNet) so that results are comparable and not leaked. Architecture-specific choices (graph edges,
node features, ESM embeddings, RSA, pLDDT/PAE, …) are computed **later** from the raw per-residue coordinates stored here.
So: yes — dataset generation does not depend on the architecture; only the feature/graph step does.

## What the pipeline does (standard recipe, source in brackets)
| Decision | Value | Source |
|---|---|---|
| Structures | PDB, X-ray/EM, resolution ≤ 3.0 Å, biological assembly 1 | PeSTo/MPBind use all assemblies, no resolution filter → our simplification (assembly 1 avoids crystal-packing contacts) |
| Cleaning | drop H/D/water/ligands, non-standard residues (MSE→MET), first altloc, require N/CA/C/O | PeSTo/MPBind |
| Interface label | residue is positive if **any heavy atom ≤ 5 Å** from a heavy atom of a *different protein chain* in the assembly | PeSTo/MPBind (5 Å heavy atom) |
| Chain filter | 48 ≤ length ≤ 1000, partner chain ≥ 5 res, ≥ 5 interface residues, < 10 % chain breaks | MPBind (min 48 res, size cap) + our QC |
| Redundancy | MMseqs2 clustering, 30 % identity, 80 % coverage; keep best-resolution chain per cluster (`members_per_cluster=1`) | PeSTo/MPBind (30 %); one-per-cluster keeps size moderate |
| Split | whole clusters → train/val/test = 70/15/15; then re-search val/test vs train and drop any ≥30 % hit | PeSTo ratios; leakage safety-net is ours |
| Temporal test | entries deposited 2022-01-01 → 2024-06-21, remove anything ≥30 % to any earlier chain, cluster, 1 per cluster | MPBind “Test2_data” |

PeSTo/MPBind train on ~376k chains (all members of each cluster). We deliberately use a much smaller **non-redundant** set
(expected order of 10⁴ chains – **not yet measured**, see status). Set `members_per_cluster` in `ppi_prep/config.py` to grow it.

All constants are in `ppi_prep/config.py`.

## Layout
```
preprocessing/
  context.md                 <- this file
  run_all.sh                 <- runs steps 01-05
  01_select_entries.py       wwPDB entries.idx -> data/entries.tsv (resolution/method/date, era pre|post)
  02_select_candidates.py    CLUSTER-FIRST: cluster all SEQRES sequences (30 %), greedy set cover -> data/selected_entries.tsv
                             (only these entries get downloaded, instead of the whole filtered PDB)
  02_download_assemblies.py  assembly1 mmCIF from files.wwpdb.org for selected entries (parallel, resumable)
  03_extract_chains.py       clean chains + 5 Å labels -> data/chains/<pdb>_<chain>.npz, data/chains.tsv
                             (--local DIR: run on any local pdb/cif files)
  04_cluster_split.py        MMseqs2 cluster, leakage-free splits -> data/splits.tsv, data/{train,val,test,test_temporal}.txt
  05_dataset_stats.py        sanity report
  ppi_prep/                  structure.py (labels), mmseqs.py, net.py, config.py
  tests/test_labels.py       pytest: cutoff geometry, virtual CB, 1BRS barnase–barstar
  data/                      generated, git-ignored
```
Per-chain `.npz` (architecture-agnostic): `seq`, `resseq`, `icode`, `bb` (L,4,3: N,CA,C,O), `cb` (L,3; virtual CB for Gly), `sc`
(L,3; side-chain centroid), `label` (L,) int8.
Usage: `pip install -r requirements.txt`, install MMseqs2 (`conda install -c bioconda mmseqs2`), then `./run_all.sh`.

## Cluster-first download
Downloading every filtered PDB entry (probably >100k) is wasteful because step 4 keeps ~1 chain per cluster anyway. Step 02a therefore
clusters the SEQRES sequences first and picks, by greedy set cover, the fewest entries (preferring multi-chain entries, then better
resolution) so each 30 % cluster is covered by `cover_per_cluster` (=2) entries. Chains in a downloaded entry that belong to other
clusters are kept too; step 4 does the final dedup on the modelled chains, so the 02a selection can only affect cost, not the leakage guarantees.
Post-era sequences similar to any pre-era sequence are skipped already here.
Caveat: SEQRES chain IDs are not used downstream (assembly files use label_asym_id), selection is by entry only.

## Running on Kaggle
Internet must be ON (Settings sidebar). Clone with `git clone --depth 1 --filter=blob:none --sparse -b claude/preprocessing-pipeline <repo>`
then `git sparse-checkout set preprocessing`; `pip install biopython scipy pandas requests tqdm pytest`; download the static MMseqs2 binary
(`https://github.com/soedinglab/MMseqs2/releases/latest/download/mmseqs-linux-avx2.tar.gz`) and put `mmseqs/bin` on PATH; then `bash run_all.sh`.
Save `data/` (without `data/raw`) as a Kaggle Dataset / zip before the session ends.

## Status (honest)
- Code written; unit tests pass (8/8, incl. the cluster-first set cover). `02_select_candidates.py` verified on a synthetic seqres + entries table with real MMseqs2 (41 clusters covered twice by 41 entries instead of ~82). On `code/1BRS.pdb` the labels give ~19–20 interface residues per barstar chain, matching the
  literature for barnase–barstar. mmCIF (.cif.gz) parsing path verified on a converted copy of 1BRS.
- `04_cluster_split.py` verified end-to-end with real MMseqs2 on a **synthetic** table (homolog families + post-era chains): families
  never straddle splits, homologous post-era chains removed.
- **NOT yet run on the real PDB**: the sandbox this was written in cannot reach wwPDB/RCSB. Steps 01, 02a (seqres download) and 02b (assembly download) are therefore
  untested against the live servers, and the final dataset size / positive rate are unknown. First thing to do on a machine with internet:
  run `./run_all.sh` (try `python 02_download_assemblies.py --limit 200` first), inspect `data/stats.tsv`.
- The older `code/pipeline.py` (CA–CA 8 Å labels, toy graph) is superseded for labelling: standard is 5 Å heavy-atom.

## Known simplifications / things to revisit
- Assembly 1 only (PeSTo uses all). Date uses deposition date from `entries.idx`, not release date.
- Labels are *unpartnered* (any other protein chain). Partner-specific labels can be re-derived from the assemblies (cached in `data/raw`).
- MMseqs2 greedy clustering is approximate; hence the explicit train-vs-val/test re-search.
- Large complexes (> 250k atoms) are skipped.

## Next steps
1. Run on real data on a networked machine; check size (target roughly 5–15k chains), positive rate (PPI typically ~10–20 % of residues),
   split sizes; adjust `max_resolution` / `members_per_cluster` if needed.
2. Feature stage (separate step, reads the `.npz` files): ESM-2 (or ProtTrans) embeddings, RSA/SASA, DSSP, torsions, graph (kNN/radius,
   e.g. MPBind uses CA ≤ 15 Å).
3. Baselines before the novel model: sequence-only (ESM + MLP), then ESM + plain geometric GNN (this is “M0” in the proposal ladder).
   Metrics: PR-AUC (primary), ROC-AUC, validation-selected MCC; also evaluate on `test_temporal`.
4. Only then: the novelty (confidence/pLDDT/PAE gating, biophysical priors). That needs a *paired AlphaFold-structure test set*
   (UniProt-mapped via SIFTS) — a later extension of this pipeline, not needed for the baseline dataset.

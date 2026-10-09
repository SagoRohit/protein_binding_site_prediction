#!/usr/bin/env bash
# Full dataset build. Needs internet (files.wwpdb.org) and `mmseqs` on PATH.
set -euo pipefail
cd "$(dirname "$0")"
python 01_select_entries.py                  # filter PDB index (resolution, method, date)
python 02_select_candidates.py --threads 4   # cluster-first: choose the few entries worth downloading
python 02_download_assemblies.py --workers 8 # download only those
python 03_extract_chains.py                  # clean chains + 5 A interface labels
python 04_cluster_split.py --threads 4       # final 30% redundancy removal + splits
python 05_dataset_stats.py

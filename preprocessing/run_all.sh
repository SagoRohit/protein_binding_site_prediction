#!/usr/bin/env bash
# Full dataset build. Needs internet (files.wwpdb.org) and `mmseqs` on PATH.
set -euo pipefail
cd "$(dirname "$0")"
python 01_select_entries.py
python 02_download_assemblies.py --workers 8
python 03_extract_chains.py
python 04_cluster_split.py --threads 8
python 05_dataset_stats.py

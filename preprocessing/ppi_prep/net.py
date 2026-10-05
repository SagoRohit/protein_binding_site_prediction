import os
import time

import requests

WWPDB = "https://files.wwpdb.org/pub/pdb"


def fetch(url, dest, retries=4):
    """Download url -> dest (atomic, skips existing). Returns True on success."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return True
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    for k in range(retries):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 404:
                return False
            r.raise_for_status()
            with open(dest + ".part", "wb") as fh:
                fh.write(r.content)
            os.replace(dest + ".part", dest)
            return True
        except requests.RequestException:
            time.sleep(2 ** (k + 1))
    return False


def assembly_url(pdb_id, assembly_id=1):
    p = pdb_id.lower()
    return f"{WWPDB}/data/assemblies/mmCIF/divided/{p[1:3]}/{p}-assembly{assembly_id}.cif.gz"

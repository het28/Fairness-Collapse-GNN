"""
Prepare GNNs-FAME datasets so the loader finds expected files.
- German: repo has german_edges.txt but loader expects german_edges.csv with columns uid1, uid2.
  We create german_edges.csv from german_edges.txt (space-separated, no header).
- Credit: unzip credit.zip into dataset/credit/ if credit.csv is missing.
Run from repo root: python3 scripts/prepare_gnns_fame_data.py
"""

import os
import shutil
import zipfile
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent.parent
    for folder in ("GNNs-FAME-main", "GNNs-FAME"):
        repo = root / folder
        if not repo.is_dir():
            continue
        # German: german_edges.txt -> german_edges.csv with uid1, uid2
        german_dir = repo / "dataset" / "german"
        edges_txt = german_dir / "german_edges.txt"
        edges_csv = german_dir / "german_edges.csv"
        if edges_txt.exists() and not edges_csv.exists():
            with open(edges_txt) as f:
                lines = [l.strip().split() for l in f if l.strip()]
            with open(edges_csv, "w") as f:
                f.write("uid1,uid2\n")
                for parts in lines:
                    if len(parts) >= 2:
                        f.write(f"{parts[0]},{parts[1]}\n")
            print(f"Created {edges_csv}")
        # Credit: unzip if needed; loader expects credit.csv and credit_edges.csv in dataset/credit/
        credit_dir = repo / "dataset" / "credit"
        credit_zip = credit_dir / "credit.zip"
        credit_csv = credit_dir / "credit.csv"
        if credit_zip.exists() and not credit_csv.exists():
            with zipfile.ZipFile(credit_zip) as z:
                z.extractall(credit_dir)
            print(f"Extracted {credit_zip}")
        # If zip created a nested credit/ folder, move files up
        inner = credit_dir / "credit"
        if inner.is_dir() and not credit_csv.exists():
            for f in inner.iterdir():
                dest = credit_dir / f.name
                if not dest.exists() or f.stat().st_mtime > dest.stat().st_mtime:
                    shutil.copy2(f, dest)
            print(f"Copied files from {inner} to {credit_dir}")
        # credit_edges: loader expects .csv with uid1, uid2
        edges_txt = credit_dir / "credit_edges.txt"
        edges_csv = credit_dir / "credit_edges.csv"
        if edges_txt.exists() and not edges_csv.exists():
            with open(edges_txt) as f:
                lines = [l.strip().split() for l in f if l.strip()]
            with open(edges_csv, "w") as f:
                f.write("uid1,uid2\n")
                for parts in lines:
                    if len(parts) >= 2:
                        f.write(f"{parts[0]},{parts[1]}\n")
            print(f"Created {edges_csv}")
        break
    else:
        print("GNNs-FAME-main/ or GNNs-FAME/ not found. Run from gnn-fairness-uncertainty root.")


if __name__ == "__main__":
    main()

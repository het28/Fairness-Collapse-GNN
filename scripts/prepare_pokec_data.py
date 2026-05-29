#!/usr/bin/env python3
"""
Prepare Pokec data for GNNs-FAME (pokec-z, pokec-n).
- If dataset/pokec/pokec.zip is Git LFS, run 'git lfs pull' then unzip.
- Unzip produces region_job*.csv and/or region_job*_relationship.txt.
- Convert .txt edge files to .csv (uid1, uid2) via set_uid or this script.
Required for pokec-z: region_job.csv, region_job_relationship.csv (or .txt).
Required for pokec-n: region_job_2.csv, region_job_2_relationship.csv (or .txt).
"""
import os
import subprocess
import sys
from pathlib import Path

def main():
    repo_root = Path(__file__).resolve().parent.parent
    # GNNs-FAME dataset path
    pokec_dir = repo_root / "GNNs-FAME-main" / "dataset" / "pokec"
    if not pokec_dir.is_dir():
        pokec_dir = repo_root / "GNNs-FAME" / "dataset" / "pokec"
    if not pokec_dir.is_dir():
        print("GNNs-FAME-main/dataset/pokec not found.", file=sys.stderr)
        sys.exit(1)

    zip_path = pokec_dir / "pokec.zip"
    if zip_path.is_file():
        size = zip_path.stat().st_size
        if size < 500:
            print("pokec.zip looks like a Git LFS pointer. Pulling LFS...")
            try:
                subprocess.run(["git", "lfs", "pull"], cwd=repo_root, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                print("Git LFS pull failed or git not found:", e)
                print("Run manually: git lfs pull  (in project root), then re-run this script.")
                sys.exit(1)
        if zip_path.stat().st_size > 500:
            print("Unzipping pokec.zip...")
            subprocess.run(["unzip", "-o", str(zip_path)], cwd=str(pokec_dir), check=True)

    # Convert .txt edge files to .csv if needed (preprocess_data also accepts .txt now)
    for name, txt_name in [
        ("region_job_relationship", "region_job_relationship.txt"),
        ("region_job_2_relationship", "region_job_2_relationship.txt"),
    ]:
        txt_path = pokec_dir / txt_name
        csv_path = pokec_dir / (name + ".csv")
        if txt_path.is_file() and not csv_path.is_file():
            print(f"Converting {txt_name} -> {name}.csv (space-separated uid1 uid2)")
            with open(txt_path) as f:
                lines = [l.split() for l in f if len(l.split()) >= 2]
            with open(csv_path, "w") as f:
                f.write("uid1,uid2\n")
                for parts in lines:
                    f.write(f"{parts[0]},{parts[1]}\n")

    # Check required files
    ok = True
    for region, (node_name, edge_csv, edge_txt) in [
        ("pokec-z", ("region_job.csv", "region_job_relationship.csv", "region_job_relationship.txt")),
        ("pokec-n", ("region_job_2.csv", "region_job_2_relationship.csv", "region_job_2_relationship.txt")),
    ]:
        node_file = pokec_dir / node_name
        has_edges = (pokec_dir / edge_csv).is_file() or (pokec_dir / edge_txt).is_file()
        if not node_file.is_file():
            print(f"Missing node file for {region}: {node_name}. Obtain from SNAP Pokec or GNNs-FAME data.", file=sys.stderr)
            ok = False
        if not has_edges:
            print(f"Missing edge file for {region} ({edge_csv} or {edge_txt}). Run after unzip or obtain.", file=sys.stderr)
            ok = False
    if ok:
        print("Pokec data ready for pokec-z and/or pokec-n.")
    else:
        print("See GNNs-FAME README and https://snap.stanford.edu/data/soc-Pokec.html for data sources.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

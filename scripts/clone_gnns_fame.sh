#!/usr/bin/env bash
# Clone GNNs-FAME repo for uncertainty-in-fairness experiments.
# Run from repo root: ./scripts/clone_gnns_fame.sh
# GitHub may create GNNs-FAME-main when downloading; the code checks both names.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
if [[ -d "GNNs-FAME-main" ]]; then
  echo "GNNs-FAME-main/ already exists. Pulling latest."
  (cd GNNs-FAME-main && git pull)
elif [[ -d "GNNs-FAME" ]]; then
  echo "GNNs-FAME/ already exists. Pulling latest."
  (cd GNNs-FAME && git pull)
else
  git clone https://github.com/HannanJaved/GNNs-FAME.git GNNs-FAME-main
  echo "Cloned into GNNs-FAME-main/"
fi
echo "Done. Code auto-detects GNNs-FAME-main/ or GNNs-FAME/."

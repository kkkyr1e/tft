#!/usr/bin/env bash
# Fetch the Set 4 simulator at the commit this harness was validated against.
# The simulator is our fork of silverlight6/TFTMuZeroAgent: branch `develop` carries the bug
# fixes listed in its FORK_NOTES.md (each one also a separate branch for an upstream PR).
set -euo pipefail
cd "$(dirname "$0")/.."
SIM_REPO=https://github.com/kkkyr1e/TFTMuZeroAgent
SIM_COMMIT=5e2cb1cfc5e5ceb009b3e627070c00f5c6a4db47  # develop
mkdir -p third_party
if [ ! -d third_party/TFTMuZeroAgent ]; then
  git clone "$SIM_REPO" third_party/TFTMuZeroAgent
fi
git -C third_party/TFTMuZeroAgent remote set-url origin "$SIM_REPO"  # older checkouts point at upstream
git -C third_party/TFTMuZeroAgent fetch --quiet origin
git -C third_party/TFTMuZeroAgent checkout --quiet "$SIM_COMMIT"
python -m pip install "numpy>=2.0" "pettingzoo>=1.27" "gymnasium>=1.3" pytest
echo
echo "Done. Before running anything:"
echo "  export PYTHONPATH=\$PWD/third_party/TFTMuZeroAgent:\$PWD"

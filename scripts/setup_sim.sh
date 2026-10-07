#!/usr/bin/env bash
# Fetch the Set 4 simulator at the commit this harness was validated against.
set -euo pipefail
cd "$(dirname "$0")/.."
SIM_COMMIT=a5718dd55658271fbcc5bbc247345f2f55b68fe2
mkdir -p third_party
if [ ! -d third_party/TFTMuZeroAgent ]; then
  git clone https://github.com/silverlight6/TFTMuZeroAgent third_party/TFTMuZeroAgent
fi
git -C third_party/TFTMuZeroAgent fetch --quiet origin
git -C third_party/TFTMuZeroAgent checkout --quiet "$SIM_COMMIT"
python -m pip install "numpy>=2.0" "pettingzoo>=1.27" "gymnasium>=1.3" pytest
echo
echo "Done. Before running anything:"
echo "  export PYTHONPATH=\$PWD/third_party/TFTMuZeroAgent:\$PWD"

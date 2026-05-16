#!/usr/bin/env bash
# Run once on the cloud instance after SSH-ing in.
# Installs acids-rave and applies scipy/numpy compatibility patches.
set -euo pipefail

WORKSPACE=${WORKSPACE:-/workspace/lambertify}
cd "$WORKSPACE"

echo "==> pip: upgrading and installing acids-rave..."
pip install --upgrade pip
# PyTorch is already in the Vast.ai base image; acids-rave just needs the rave CLI
pip install acids-rave

echo "==> Applying pqmf.py compatibility patches..."
python3 cloud/patch_pqmf.py
echo ""

echo "==> Verifying rave binary..."
rave --help > /dev/null && echo "    rave CLI: OK"
echo ""

echo "Done. Run cloud/train.sh to start preprocessing + training."

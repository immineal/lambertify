#!/usr/bin/env bash
# Transfer data and results between local machine and Vast.ai instance.
#
# Set HOST and PORT from the Vast.ai dashboard (shown as the SSH command).
# Example: HOST=ssh3.vast.ai PORT=12345 ./cloud/sync.sh push
#
# Commands:
#   push   — upload MP3s + cloud scripts to the instance
#   pull   — download trained checkpoints and exported .ts model
#   status — show what's in runs/ on the remote

set -euo pipefail

HOST=${HOST:?Set HOST to your Vast.ai SSH address (e.g. ssh3.vast.ai)}
PORT=${PORT:?Set PORT to your Vast.ai SSH port (e.g. 12345)}
REMOTE_USER=${REMOTE_USER:-root}
REMOTE_DIR=${REMOTE_DIR:-/workspace/lambertify}
LOCAL_DIR=$(cd "$(dirname "$0")/.." && pwd)

SSH="ssh -p $PORT -o StrictHostKeyChecking=no"
RSYNC="rsync -avz --progress -e \"ssh -p $PORT -o StrictHostKeyChecking=no\""

cmd=${1:-help}

case "$cmd" in

push)
    echo "==> Creating remote workspace..."
    $SSH $REMOTE_USER@$HOST "mkdir -p $REMOTE_DIR/data $REMOTE_DIR/cloud $REMOTE_DIR/processed"

    echo "==> Uploading MP3 data (~790MB)..."
    eval "$RSYNC \"$LOCAL_DIR/data/\" $REMOTE_USER@$HOST:$REMOTE_DIR/data/"

    echo "==> Uploading cloud scripts..."
    eval "$RSYNC \"$LOCAL_DIR/cloud/\" $REMOTE_USER@$HOST:$REMOTE_DIR/cloud/"

    chmod +x "$LOCAL_DIR/cloud/"*.sh 2>/dev/null || true
    $SSH $REMOTE_USER@$HOST "chmod +x $REMOTE_DIR/cloud/*.sh"

    echo ""
    echo "Upload done. SSH in and run:"
    echo "  ssh -p $PORT $REMOTE_USER@$HOST"
    echo "  cd $REMOTE_DIR && bash cloud/setup.sh && bash cloud/train.sh"
    ;;

pull)
    echo "==> Downloading run checkpoints and exported models..."
    mkdir -p "$LOCAL_DIR/runs"
    # Pull all checkpoint files and exported .ts models, skip bulk tensorboard events
    eval "$RSYNC \
        --include='*/' \
        --include='*.ckpt' \
        --include='*.ts' \
        --include='config.gin' \
        --include='hparams.yaml' \
        --exclude='events.out.*' \
        --exclude='*.mdb' \
        $REMOTE_USER@$HOST:$REMOTE_DIR/runs/ \
        \"$LOCAL_DIR/runs/\""

    echo ""
    echo "==> Downloaded to $LOCAL_DIR/runs/"
    ls "$LOCAL_DIR/runs/"
    ;;

pull-all)
    echo "==> Downloading everything including tensorboard logs..."
    mkdir -p "$LOCAL_DIR/runs"
    eval "$RSYNC $REMOTE_USER@$HOST:$REMOTE_DIR/runs/ \"$LOCAL_DIR/runs/\""
    ;;

status)
    echo "==> Remote runs/ directory:"
    $SSH $REMOTE_USER@$HOST "
        find $REMOTE_DIR/runs -name '*.ckpt' -o -name '*.ts' 2>/dev/null \
        | while read f; do
            echo \"\$(ls -lh \$f | awk '{print \$5}')  \$f\"
        done | sort
        echo ''
        echo 'Latest checkpoint:'
        find $REMOTE_DIR/runs -name 'last*.ckpt' | sort -t_ -k1 | tail -1 || echo '  (none yet)'
    "
    ;;

help|*)
    echo "Usage: HOST=... PORT=... ./cloud/sync.sh <push|pull|pull-all|status>"
    echo ""
    echo "  push      — upload data + scripts to remote"
    echo "  pull      — download checkpoints + .ts model (no tensorboard blobs)"
    echo "  pull-all  — download entire runs/ directory"
    echo "  status    — list checkpoints and .ts files on remote"
    ;;

esac

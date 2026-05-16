#!/usr/bin/env bash
# RAVE v1 training script for cloud GPU.
# Preprocesses data (262144 samples = ~6s context) then trains for 2M steps.
#
# Tunable via env vars:
#   NAME, N_SIGNAL, BATCH, WORKERS, STEPS, VAL_EVERY
#   DATA_DIR, PROCESSED_DIR
#
# Defaults are set for A10G/L4 24GB with 8 CPU workers.
set -euo pipefail

NAME=${NAME:-lambert-v1}
N_SIGNAL=${N_SIGNAL:-262144}   # 262144 / 44100 ≈ 5.9s context
BATCH=${BATCH:-16}
WORKERS=${WORKERS:-8}
STEPS=${STEPS:-2000000}
VAL_EVERY=${VAL_EVERY:-50000}  # save checkpoint every 50k steps (40 total)
SAMPLE_RATE=${SAMPLE_RATE:-44100}

WORKSPACE=${WORKSPACE:-/workspace/lambertify}
DATA_DIR=${DATA_DIR:-$WORKSPACE/data}
PROCESSED_DIR=${PROCESSED_DIR:-$WORKSPACE/processed/rave_preprocessed}

cd "$WORKSPACE"

echo "========================================"
echo " RAVE v1 cloud training"
echo "  n_signal : $N_SIGNAL ($(python3 -c "print(f'{$N_SIGNAL/$SAMPLE_RATE:.1f}')")s)"
echo "  batch    : $BATCH"
echo "  workers  : $WORKERS"
echo "  steps    : $STEPS"
echo "  val_every: $VAL_EVERY"
echo "  data     : $DATA_DIR"
echo "========================================"
echo ""

# ---- Preprocessing ----
if [ -f "$PROCESSED_DIR/metadata.yaml" ]; then
    stored_ns=$(python3 -c "
import yaml
with open('$PROCESSED_DIR/metadata.yaml') as f:
    d = yaml.safe_load(f)
print(d.get('n_signal', 0))
" 2>/dev/null || echo 0)
    if [ "$stored_ns" = "$N_SIGNAL" ]; then
        echo "==> Preprocessed data already exists for n_signal=$N_SIGNAL, skipping."
    else
        echo "==> n_signal mismatch (stored: $stored_ns, wanted: $N_SIGNAL). Reprocessing..."
        rm -rf "$PROCESSED_DIR"
        rave preprocess \
            --input_path  "$DATA_DIR" \
            --output_path "$PROCESSED_DIR" \
            --sampling_rate $SAMPLE_RATE \
            --num_signal   $N_SIGNAL
    fi
else
    echo "==> Preprocessing audio (this takes a few minutes)..."
    rave preprocess \
        --input_path  "$DATA_DIR" \
        --output_path "$PROCESSED_DIR" \
        --sampling_rate $SAMPLE_RATE \
        --num_signal   $N_SIGNAL
fi
echo ""

# ---- Training ----
echo "==> Starting training..."
echo "    Logs: runs/${NAME}_*/version_*/events.out.tfevents.*"
echo "    Checkpoints saved every $VAL_EVERY steps."
echo "    Estimated time on A100 40GB: ~15-20h"
echo "    Estimated time on A10G 24GB: ~25-35h"
echo ""

rave train \
    --config    v1 \
    --db_path   "$PROCESSED_DIR" \
    --name      "$NAME" \
    --n_signal  $N_SIGNAL \
    --batch     $BATCH \
    --workers   $WORKERS \
    --max_steps $STEPS \
    --val_every $VAL_EVERY \
    --gpu       0

# ---- Export ----
echo ""
echo "==> Training complete. Exporting .ts model..."
RUN_DIR=$(ls -td runs/${NAME}_* 2>/dev/null | head -1)
if [ -z "$RUN_DIR" ]; then
    echo "ERROR: no run directory found for name '$NAME'" >&2
    exit 1
fi
rave export --run "$RUN_DIR"
echo ""
echo "Exported model: $(ls $RUN_DIR/*.ts 2>/dev/null | head -1)"
echo ""
echo "Run cloud/sync.sh pull to download results."

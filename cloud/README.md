# Cloud Training — RAVE v1 on Vast.ai

**Target:** RAVE v1, n_signal=262144 (~6s context), batch=16, 8 workers, 2M steps  
**GPU:** A10G or L4 (24 GB VRAM) — fits comfortably  
**Estimated cost:** ~$15-25 total (A10G at ~$0.50-0.80/hr × 25-35h)

---

## 1. Rent a Vast.ai instance

1. Go to [vast.ai](https://vast.ai) → **Search** tab
2. Filter by:
   - **GPU:** NVIDIA A10G or L4 (search "A10" or "L4")
   - **VRAM:** ≥ 24 GB
   - **Disk:** ≥ 30 GB
   - **Template image:** `pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime`
     (or any recent PyTorch + CUDA 12.x image)
3. Click **Rent** on a cheap offer (sort by $/hr)
4. Once the instance shows **Running**, click **Connect** to get the SSH command:
   ```
   ssh -p 12345 root@ssh3.vast.ai
   ```
   Note the host (`ssh3.vast.ai`) and port (`12345`) — you'll use them in every command below.

---

## 2. Upload data and scripts (run locally)

```bash
cd /path/to/lambertify

# Replace HOST and PORT with your Vast.ai values
HOST=ssh3.vast.ai PORT=12345 bash cloud/sync.sh push
```

This uploads:
- `data/` — all MP3 files (~790 MB, takes ~2-5 min depending on connection)
- `cloud/` — setup and training scripts

---

## 3. Set up the environment (run on remote)

```bash
ssh -p 12345 root@ssh3.vast.ai

cd /workspace/lambertify
bash cloud/setup.sh
```

This installs `acids-rave` and patches `pqmf.py` for modern scipy/numpy.
Takes ~2 minutes.

---

## 4. Start training (run on remote)

```bash
# Still on the remote instance:
cd /workspace/lambertify

# Recommended: run in screen/tmux so it survives SSH disconnects
screen -S rave
bash cloud/train.sh
```

To detach from screen: `Ctrl-A D`  
To re-attach later: `screen -r rave`

**What train.sh does:**
1. Preprocesses audio with n_signal=262144 (~5 min)
2. Trains RAVE v1 for 2M steps, checkpointing every 50k steps
3. Exports a `.ts` model at the end

**Estimated runtimes:**
| GPU | Steps/sec | Total time |
|-----|-----------|------------|
| L4 24GB | ~20-30 | ~20-28h |
| A10G 24GB | ~25-35 | ~16-22h |
| A100 40GB | ~50-70 | ~8-11h |

---

## 5. Monitor progress (optional)

From your local machine, watch the TensorBoard logs:

```bash
# Pull logs periodically and view locally
HOST=ssh3.vast.ai PORT=12345 bash cloud/sync.sh pull-all
tensorboard --logdir runs/
```

Or SSH in and check the screen session:
```bash
ssh -p 12345 root@ssh3.vast.ai
screen -r rave
```

---

## 6. Download results

When training finishes (or when you want an intermediate checkpoint):

```bash
# From your local machine:
HOST=ssh3.vast.ai PORT=12345 bash cloud/sync.sh pull
```

This downloads only `.ckpt`, `.ts`, `config.gin`, and `hparams.yaml` — skipping the large tensorboard event files.

The exported `.ts` model lands in `runs/lambert-v1_<hash>/lambert-v1.ts`.  
Point the Lambertify UI at it to generate audio.

---

## 7. Stop the instance

**Important:** Vast.ai bills by the hour. Stop the instance once you have your checkpoints.

Vast.ai dashboard → your running instance → **Destroy** (or **Stop** if you want to pause and resume later — stopped instances still cost a small amount for storage).

---

## Customising parameters

Override any parameter via env var before running `train.sh`:

```bash
# Example: shorter run with smaller context to verify it works
N_SIGNAL=131072 STEPS=100000 BATCH=16 bash cloud/train.sh

# Full run with even more context (needs A100 80GB for batch=16)
N_SIGNAL=524288 STEPS=2000000 BATCH=8 bash cloud/train.sh
```

| Variable | Default | Meaning |
|----------|---------|---------|
| `NAME` | `lambert-v1` | Run name prefix |
| `N_SIGNAL` | `262144` | Samples per training clip (~6s @ 44100Hz) |
| `BATCH` | `16` | Batch size |
| `WORKERS` | `8` | DataLoader workers |
| `STEPS` | `2000000` | Total training steps |
| `VAL_EVERY` | `50000` | Steps between checkpoints |

---

## Resume from a checkpoint

If training is interrupted, find the last checkpoint on the remote:

```bash
HOST=ssh3.vast.ai PORT=12345 bash cloud/sync.sh status
```

Then restart with `--ckpt`:

```bash
# Edit train.sh to add --ckpt, or run rave train directly:
rave train \
    --config v1 \
    --db_path /workspace/lambertify/processed/rave_preprocessed \
    --name lambert-v1 \
    --n_signal 262144 \
    --batch 16 --workers 8 \
    --max_steps 2000000 \
    --val_every 50000 \
    --gpu 0 \
    --ckpt runs/lambert-v1_<hash>/version_N/checkpoints/last.ckpt
```

---

## Troubleshooting

**`kaiser` import error** — Run `python3 cloud/patch_pqmf.py` again; it's idempotent.

**OOM (out of memory)** — Reduce `BATCH=8` or try a 40GB GPU.  
  ```bash
  BATCH=8 bash cloud/train.sh
  ```

**`rave` command not found** — `pip install acids-rave` didn't finish. Re-run `setup.sh`.

**Slow upload** — The 790 MB data upload can be slow on home connections. Consider using rclone with a cloud storage bucket (S3/GCS) as an intermediate if your connection is < 10 Mbps.

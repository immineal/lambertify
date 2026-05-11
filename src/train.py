"""Two-stage training pipeline.

Stage 1 – Train MusicVAE on mel-spectrogram chunks.
Stage 2 – Extract VAE latent codes, then train LatentLSTM as an
           autoregressive prior for coherent long-form generation.

Usage
-----
  python src/train.py                        # full pipeline with smart defaults
  python src/train.py --stage vae            # VAE only
  python src/train.py --stage lstm           # LSTM only (VAE must exist)
  python src/train.py --vae_epochs 50 --lstm_epochs 30 --latent_dim 256
  python src/train.py --batch_size 128 --device cuda:0
"""
import sys
import os
import json
import argparse
import time
from typing import Callable, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import torch
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm

from src.model import MusicVAE, LatentLSTM
from src.dataset import get_vae_loaders, get_lstm_loaders
from src.hardware import query_gpus, recommend_params

PROCESSED_DIR  = Path(ROOT) / "processed"
CHECKPOINT_DIR = Path(ROOT) / "checkpoints"
RUNS_DIR       = CHECKPOINT_DIR / "runs"
CHECKPOINT_DIR.mkdir(exist_ok=True)
RUNS_DIR.mkdir(exist_ok=True)

ProgressCB = Optional[Callable[[str, int, int, float, float, int], None]]
# callback signature: (stage, epoch, total_epochs, train_loss, val_loss, vram_used_mb)


def _vram_mb() -> int:
    """Current VRAM usage of the default CUDA device in MB, or 0 if unavailable."""
    if not torch.cuda.is_available():
        return 0
    try:
        return torch.cuda.memory_allocated() // (1024 * 1024)
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Stage 1 – VAE
# ---------------------------------------------------------------------------

def train_vae(
    n_epochs: int = 50,
    batch_size: int = 32,
    lr: float = 2e-4,
    latent_dim: int = 128,
    kl_max: float = 5e-4,
    kl_ramp_pct: float = 0.4,
    num_workers: int = 2,
    device: torch.device | None = None,
    progress_cb: ProgressCB = None,
    checkpoint_every: int = 50,
    run_id: str | None = None,
) -> MusicVAE:
    _run_id = run_id   # used inside loop for registry calls

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== Stage 1: VAE training on {device} ===")

    chunks_path = str(PROCESSED_DIR / "chunks.npy")
    if not Path(chunks_path).exists():
        raise FileNotFoundError(
            f"Preprocessed chunks not found at {chunks_path}.\n"
            "Run preprocessing first: python src/preprocess.py"
        )

    train_loader, val_loader = get_vae_loaders(
        chunks_path,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    print(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")

    model = MusicVAE(latent_dim=latent_dim).to(device)
    print(f"MusicVAE parameters: {model.n_params():,}")

    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr * 0.05
    )

    # Per-run checkpoint directory
    if run_id:
        run_dir = RUNS_DIR / run_id
        run_dir.mkdir(exist_ok=True)
    else:
        run_dir = CHECKPOINT_DIR

    history: dict[str, list] = {
        "train_total": [], "train_recon": [], "train_kl": [],
        "val_total":   [], "val_recon":   [],
    }
    best_val  = float("inf")
    best_ckpt: dict | None = None

    for epoch in range(1, n_epochs + 1):
        kl_weight = kl_max * min(1.0, epoch / max(1, int(n_epochs * kl_ramp_pct)))

        # ---- Train ---------------------------------------------------------
        model.train()
        t_losses: list[tuple[float, float, float]] = []
        for batch in tqdm(train_loader, desc=f"VAE {epoch:3d}/{n_epochs}", leave=False):
            batch = batch.to(device, non_blocking=True)
            recon, mu, logvar = model(batch)
            loss, recon_l, kl_l = MusicVAE.loss(recon, batch, mu, logvar, kl_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_losses.append((loss.item(), recon_l.item(), kl_l.item()))

        # ---- Validate ------------------------------------------------------
        model.eval()
        v_losses: list[tuple[float, float]] = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device, non_blocking=True)
                recon, mu, logvar = model(batch)
                loss, recon_l, _ = MusicVAE.loss(recon, batch, mu, logvar, kl_weight)
                v_losses.append((loss.item(), recon_l.item()))

        t_mean = [float(np.mean([x[i] for x in t_losses])) for i in range(3)]
        v_mean = [float(np.mean([x[i] for x in v_losses])) for i in range(2)]

        history["train_total"].append(t_mean[0])
        history["train_recon"].append(t_mean[1])
        history["train_kl"].append(t_mean[2])
        history["val_total"].append(v_mean[0])
        history["val_recon"].append(v_mean[1])

        lr_now = scheduler.get_last_lr()[0]
        scheduler.step()

        cfg_dict = {"n_mels": model.n_mels, "n_frames": model.n_frames, "latent_dim": latent_dim}

        tag = ""
        if v_mean[0] < best_val:
            best_val = v_mean[0]
            best_ckpt = {
                "epoch":               epoch,
                "model_state_dict":    model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss":            best_val,
                "config":              cfg_dict,
                "history":             history,
            }
            # Save to both legacy path (backwards compat) and per-run dir
            torch.save(best_ckpt, CHECKPOINT_DIR / "vae_best.pt")
            if run_id:
                torch.save(best_ckpt, run_dir / "vae_best.pt")
                # Register in model registry
                try:
                    from src.model_manager import registry
                    registry.register_vae_lstm(
                        run_id=run_id,
                        vae_path=str(run_dir / "vae_best.pt"),
                        lstm_path=None,
                        config=cfg_dict,
                        vae_epoch=epoch,
                        vae_val_loss=best_val,
                    )
                except Exception:
                    pass
            tag = "  ✓ best"

        # ---- Periodic checkpoint --------------------------------------------
        if checkpoint_every > 0 and epoch % checkpoint_every == 0:
            ep_ckpt = {
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "val_loss":         v_mean[0],
                "config":           cfg_dict,
            }
            ep_path = run_dir / f"vae_ep{epoch:04d}.pt"
            torch.save(ep_ckpt, ep_path)
            if run_id:
                try:
                    from src.model_manager import registry
                    registry.add_checkpoint(run_id, epoch, str(ep_path))
                except Exception:
                    pass
            tag += f"  [saved ep{epoch}]"

        vram = _vram_mb()
        print(
            f"  Ep {epoch:3d}/{n_epochs}  "
            f"train={t_mean[0]:.4f}  val={v_mean[0]:.4f}  "
            f"recon={t_mean[1]:.4f}  kl={t_mean[2]:.5f}  "
            f"lr={lr_now:.2e}  vram={vram}MB{tag}"
        )

        if progress_cb is not None:
            progress_cb("vae", epoch, n_epochs, t_mean[0], v_mean[0], vram)

    # Save final checkpoint regardless of best_val
    final_ckpt = {
        "epoch":            n_epochs,
        "model_state_dict": model.state_dict(),
        "val_loss":         v_mean[0],
        "config": {
            "n_mels": model.n_mels, "n_frames": model.n_frames,
            "latent_dim": latent_dim,
        },
        "history": history,
    }
    torch.save(final_ckpt, CHECKPOINT_DIR / "vae_final.pt")
    print(f"\nVAE done. Best val loss: {best_val:.4f}")
    return model


# ---------------------------------------------------------------------------
# Latent extraction (between stages)
# ---------------------------------------------------------------------------

def extract_latents(
    model: MusicVAE,
    device: torch.device,
    batch_size: int = 256,
    num_workers: int = 2,
) -> None:
    """Encode all training chunks into latent vectors and save to disk."""
    from src.dataset import MelChunkDataset
    from torch.utils.data import DataLoader

    chunks_path = str(PROCESSED_DIR / "chunks.npy")
    ds     = MelChunkDataset(chunks_path, augment=False)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=device.type == "cuda",
    )

    model.eval()
    all_mu: list[np.ndarray] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting latents"):
            batch = batch.to(device, non_blocking=True)
            mu, _ = model.encode(batch)
            all_mu.append(mu.cpu().numpy())

    latents = np.concatenate(all_mu, axis=0)   # (N, latent_dim)
    np.save(PROCESSED_DIR / "latents.npy", latents.astype(np.float32))
    print(f"Saved latents: {latents.shape}  ({latents.nbytes / 1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# Stage 2 – LSTM
# ---------------------------------------------------------------------------

def train_lstm(
    n_epochs: int = 30,
    batch_size: int = 128,
    seq_len: int = 16,
    lr: float = 1e-3,
    latent_dim: int = 128,
    num_workers: int = 2,
    device: torch.device | None = None,
    progress_cb: ProgressCB = None,
    checkpoint_every: int = 50,
    run_id: str | None = None,
) -> LatentLSTM:
    _run_id = run_id
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== Stage 2: LSTM training on {device} ===")

    latents_path    = str(PROCESSED_DIR / "latents.npy")
    song_index_path = str(PROCESSED_DIR / "song_index.npy")

    for p in (latents_path, song_index_path):
        if not Path(p).exists():
            raise FileNotFoundError(
                f"Required file not found: {p}\n"
                "Run Stage 1 first: python src/train.py --stage vae"
            )

    train_loader, val_loader = get_lstm_loaders(
        latents_path, song_index_path,
        batch_size=batch_size, seq_len=seq_len,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    print(f"LSTM sequences — train: {len(train_loader)}  val: {len(val_loader)}")

    model = LatentLSTM(latent_dim=latent_dim).to(device)
    print(f"LatentLSTM parameters: {model.n_params():,}")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr * 0.05
    )

    history: dict[str, list] = {"train": [], "val": []}
    best_val  = float("inf")

    for epoch in range(1, n_epochs + 1):
        # ---- Train ---------------------------------------------------------
        model.train()
        t_losses: list[float] = []
        for x, y in tqdm(train_loader, desc=f"LSTM {epoch:3d}/{n_epochs}", leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x)
            loss = LatentLSTM.loss(pred, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_losses.append(loss.item())

        # ---- Validate ------------------------------------------------------
        model.eval()
        v_losses: list[float] = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                pred = model(x)
                v_losses.append(LatentLSTM.loss(pred, y).item())

        t_loss = float(np.mean(t_losses))
        v_loss = float(np.mean(v_losses))
        history["train"].append(t_loss)
        history["val"].append(v_loss)

        lr_now = scheduler.get_last_lr()[0]
        scheduler.step()

        tag = ""
        if v_loss < best_val:
            best_val  = v_loss
            lstm_ckpt = {
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "val_loss":         best_val,
                "config":           {"latent_dim": latent_dim, "seq_len": seq_len},
                "history":          history,
            }
            torch.save(lstm_ckpt, CHECKPOINT_DIR / "lstm_best.pt")
            if _run_id:
                lstm_path = RUNS_DIR / _run_id / "lstm_best.pt"
                torch.save(lstm_ckpt, lstm_path)
                try:
                    from src.model_manager import registry
                    registry.register_vae_lstm(
                        run_id=_run_id,
                        vae_path=str(RUNS_DIR / _run_id / "vae_best.pt"),
                        lstm_path=str(lstm_path),
                        config={"latent_dim": latent_dim},
                        vae_epoch=0,
                        vae_val_loss=0.0,
                    )
                except Exception:
                    pass
            tag = "  ✓ best"

        vram = _vram_mb()
        print(
            f"  Ep {epoch:3d}/{n_epochs}  "
            f"train={t_loss:.5f}  val={v_loss:.5f}  "
            f"lr={lr_now:.2e}  vram={vram}MB{tag}"
        )

        if progress_cb is not None:
            progress_cb("lstm", epoch, n_epochs, t_loss, v_loss, vram)

    print(f"\nLSTM done. Best val loss: {best_val:.5f}")
    return model


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_training(
    stage: str = "both",
    vae_epochs: int = 50,
    lstm_epochs: int = 30,
    vae_batch_size: int = 32,
    lstm_batch_size: int = 128,
    latent_dim: int = 128,
    lr_vae: float = 2e-4,
    lr_lstm: float = 1e-3,
    kl_max: float = 5e-4,
    kl_ramp_pct: float = 0.4,
    seq_len: int = 16,
    num_workers: int = 2,
    device_str: str | None = None,
    progress_cb: ProgressCB = None,
    checkpoint_every: int = 50,
    run_id: str | None = None,
) -> None:
    """Run the full training pipeline (or a single stage)."""
    import time as _time
    if run_id is None:
        run_id = _time.strftime("%Y%m%d_%H%M%S")

    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    if stage in ("vae", "both"):
        vae = train_vae(
            n_epochs=vae_epochs,
            batch_size=vae_batch_size,
            lr=lr_vae,
            latent_dim=latent_dim,
            kl_max=kl_max,
            kl_ramp_pct=kl_ramp_pct,
            num_workers=num_workers,
            device=device,
            progress_cb=progress_cb,
            checkpoint_every=checkpoint_every,
            run_id=run_id,
        )

    if stage in ("lstm", "both"):
        if stage == "lstm":
            # Load VAE from checkpoint
            ckpt = torch.load(CHECKPOINT_DIR / "vae_best.pt", map_location=device, weights_only=False)
            cfg  = ckpt["config"]
            vae  = MusicVAE(n_mels=cfg["n_mels"], n_frames=cfg["n_frames"],
                            latent_dim=cfg["latent_dim"]).to(device)
            vae.load_state_dict(ckpt["model_state_dict"])
            vae.eval()

        if not (PROCESSED_DIR / "latents.npy").exists():
            print("\nExtracting latent codes from training data…")
            extract_latents(vae, device, num_workers=num_workers)

        train_lstm(
            n_epochs=lstm_epochs,
            batch_size=lstm_batch_size,
            seq_len=seq_len,
            lr=lr_lstm,
            latent_dim=latent_dim,
            num_workers=num_workers,
            device=device,
            progress_cb=progress_cb,
            checkpoint_every=checkpoint_every,
            run_id=run_id,
        )

    print("\nAll done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Lambertify training pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--stage", choices=["vae", "lstm", "both"], default="both",
                   help="Which stage(s) to run")

    gpu_group = p.add_argument_group("Hardware")
    gpu_group.add_argument("--device", default=None,
                           help="PyTorch device string, e.g. cuda:0, cpu. "
                                "Defaults to auto-detect.")
    gpu_group.add_argument("--num_workers", type=int, default=None,
                           help="DataLoader worker processes. Defaults to auto.")

    vae_group = p.add_argument_group("VAE")
    vae_group.add_argument("--vae_epochs",    type=int,   default=None)
    vae_group.add_argument("--vae_batch",     type=int,   default=None, dest="vae_batch_size")
    vae_group.add_argument("--latent_dim",    type=int,   default=None)
    vae_group.add_argument("--lr_vae",        type=float, default=None)
    vae_group.add_argument("--kl_max",        type=float, default=None)

    lstm_group = p.add_argument_group("LSTM")
    lstm_group.add_argument("--lstm_epochs",  type=int,   default=None)
    lstm_group.add_argument("--lstm_batch",   type=int,   default=None, dest="lstm_batch_size")
    lstm_group.add_argument("--seq_len",      type=int,   default=None)
    lstm_group.add_argument("--lr_lstm",      type=float, default=None)

    p.add_argument("--smart_defaults", action="store_true",
                   help="Use GPU-aware recommended defaults (printed and applied)")
    return p


if __name__ == "__main__":
    args = _build_argparser().parse_args()

    # Start with GPU-aware smart defaults, then overlay any explicit CLI args
    gpus    = query_gpus()
    rec     = recommend_params(gpus[0] if gpus else None)

    if args.smart_defaults:
        print("=== Smart defaults based on hardware ===")
        for k, v in rec.items():
            if k not in ("warnings", "rationale"):
                print(f"  {k}: {v}")
        for w in rec.get("warnings", []):
            print(f"  WARNING: {w}")
        print()

    def _get(attr, key):
        return getattr(args, attr) if getattr(args, attr) is not None else rec[key]

    run_training(
        stage          = args.stage,
        vae_epochs     = _get("vae_epochs",    "vae_epochs"),
        lstm_epochs    = _get("lstm_epochs",   "lstm_epochs"),
        vae_batch_size = _get("vae_batch_size","vae_batch_size"),
        lstm_batch_size= _get("lstm_batch_size","lstm_batch_size"),
        latent_dim     = _get("latent_dim",    "latent_dim"),
        lr_vae         = _get("lr_vae",        "lr_vae"),
        lr_lstm        = _get("lr_lstm",       "lr_lstm"),
        kl_max         = _get("kl_max",        "kl_max"),
        seq_len        = _get("seq_len",       "seq_len"),
        num_workers    = args.num_workers or rec["num_workers"],
        device_str     = args.device or rec["device"],
    )

"""Audio generation from trained VAE + LSTM checkpoints.

Pipeline
--------
1. LSTM generates a sequence of latent codes with temporal coherence.
   (Falls back to spherical interpolation if LSTM not trained yet.)
2. VAE decoder maps each latent code → mel-spectrogram chunk.
3. Chunks are crossfaded into a continuous spectrogram.
4. Griffin-Lim vocoder converts the spectrogram to a waveform.
5. Result is written as a WAV file.
"""
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import torch
import soundfile as sf
from pathlib import Path

from src.model import MusicVAE, LatentLSTM
from src.audio_utils import (
    denormalize, melspec_to_audio,
    SAMPLE_RATE, HOP_LENGTH, FRAMES_PER_CHUNK, CHUNK_DURATION,
)

CHECKPOINT_DIR = Path(ROOT) / "checkpoints"
OUTPUT_DIR     = Path(ROOT) / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Checkpoint loaders
# ---------------------------------------------------------------------------

def _load_vae(device: torch.device) -> MusicVAE:
    path = CHECKPOINT_DIR / "vae_best.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"VAE checkpoint not found at {path}.\n"
            "Train first: python src/train.py --stage vae"
        )
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    cfg   = ckpt["config"]
    model = MusicVAE(
        n_mels=cfg["n_mels"],
        n_frames=cfg["n_frames"],
        latent_dim=cfg["latent_dim"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _load_lstm(device: torch.device) -> LatentLSTM | None:
    path = CHECKPOINT_DIR / "lstm_best.pt"
    if not path.exists():
        return None
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    cfg   = ckpt["config"]
    model = LatentLSTM(latent_dim=cfg["latent_dim"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slerp(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
    """Spherical linear interpolation between two latent vectors."""
    a = a / a.norm().clamp(min=1e-8)
    b = b / b.norm().clamp(min=1e-8)
    dot = (a * b).sum().clamp(-1.0, 1.0)
    theta = torch.acos(dot) * t
    perp  = (b - a * dot)
    perp  = perp / perp.norm().clamp(min=1e-8)
    return a * torch.cos(theta) + perp * torch.sin(theta)


def _crossfade_mel(mels: list[np.ndarray], overlap_frames: int = 17) -> np.ndarray:
    """Crossfade mel spectrogram chunks along the time axis.

    Joining spectrograms before Griffin-Lim (rather than crossfading audio
    after) gives phase continuity across chunk boundaries, eliminating the
    characteristic phase-incoherence artefacts.
    """
    if len(mels) == 1:
        return mels[0].copy()
    overlap = min(overlap_frames, mels[0].shape[1] // 2)
    fade_out = np.linspace(1.0, 0.0, overlap)[np.newaxis, :]   # (1, overlap)
    fade_in  = np.linspace(0.0, 1.0, overlap)[np.newaxis, :]
    result   = mels[0].copy()
    for nxt in mels[1:]:
        if result.shape[1] < overlap or nxt.shape[1] < overlap:
            result = np.concatenate([result, nxt], axis=1)
            continue
        result[:, -overlap:] = result[:, -overlap:] * fade_out + nxt[:, :overlap] * fade_in
        result = np.concatenate([result, nxt[:, overlap:]], axis=1)
    return result


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

def generate_audio(
    duration_s: float = 20.0,
    output_path: str | None = None,
    temperature: float = 1.0,
    griffin_lim_iters: int = 64,
    seed: int | None = None,
) -> str:
    """Generate *duration_s* seconds of audio. Returns the path to the WAV file."""
    duration_s = float(np.clip(duration_s, 1.0, 300.0))

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae    = _load_vae(device)
    lstm   = _load_lstm(device)

    latent_dim = vae.latent_dim
    n_chunks   = max(2, int(np.ceil(duration_s / CHUNK_DURATION)) + 1)

    # ---- Generate latent codes -------------------------------------------
    if lstm is not None:
        latents = lstm.generate(
            n_steps=n_chunks, device=str(device), temperature=temperature
        )  # (n_chunks, latent_dim)
    else:
        # No LSTM: spherical interpolation between random anchor points
        n_anchors = max(2, n_chunks // 8 + 2)
        anchors   = [
            torch.randn(latent_dim, device=device) * temperature
            for _ in range(n_anchors)
        ]
        codes: list[torch.Tensor] = []
        for t_global in np.linspace(0, n_anchors - 1, n_chunks):
            seg_i = min(int(t_global), n_anchors - 2)
            codes.append(_slerp(anchors[seg_i], anchors[seg_i + 1], t_global - int(t_global)))
        latents = torch.stack(codes)   # (n_chunks, latent_dim)

    # ---- Decode latents → mel spectrogram chunks -------------------------
    all_mels: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(n_chunks):
            z      = latents[i].unsqueeze(0).to(device)
            mel    = vae.decode(z).squeeze().cpu().numpy()   # (N_MELS, FRAMES)
            all_mels.append(denormalize(mel))

    # ---- Crossfade spectrograms first, then single Griffin-Lim pass ------
    # Running Griffin-Lim on each chunk independently and crossfading the
    # resulting audio causes phase discontinuities at chunk boundaries.
    # Joining spectrograms before vocodering gives coherent phase across the
    # whole piece.
    overlap_frames = max(8, int(0.4 * SAMPLE_RATE / HOP_LENGTH))
    mel_full = _crossfade_mel(all_mels, overlap_frames)
    audio    = melspec_to_audio(mel_full, n_iter=griffin_lim_iters)

    # ---- Trim to requested length ----------------------------------------
    target = int(duration_s * SAMPLE_RATE)
    audio  = audio[:target] if len(audio) >= target else np.pad(audio, (0, target - len(audio)))
    audio  = np.clip(audio / (np.abs(audio).max() + 1e-8), -1.0, 1.0)

    # ---- Write WAV -------------------------------------------------------
    if output_path is None:
        import time as _time
        output_path = str(OUTPUT_DIR / f"gen_{int(_time.time())}.wav")
    sf.write(output_path, audio, SAMPLE_RATE, subtype="PCM_16")
    print(f"Wrote {duration_s:.1f}s audio → {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Generate audio with Lambertify",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--duration",     type=float, default=20.0,  help="Duration in seconds")
    p.add_argument("--temperature",  type=float, default=1.0,   help="Sampling temperature")
    p.add_argument("--gl_iters",     type=int,   default=64,    help="Griffin-Lim iterations")
    p.add_argument("--output",       type=str,   default=None,  help="Output WAV path")
    p.add_argument("--seed",         type=int,   default=None,  help="Random seed")
    args = p.parse_args()

    path = generate_audio(
        duration_s=args.duration,
        output_path=args.output,
        temperature=args.temperature,
        griffin_lim_iters=args.gl_iters,
        seed=args.seed,
    )
    print(f"Done: {path}")

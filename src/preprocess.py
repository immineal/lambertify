"""Preprocess all MP3 files into normalised mel-spectrogram chunks.

Outputs
-------
processed/chunks.npy        float32 (N, N_MELS, FRAMES_PER_CHUNK)
processed/song_index.npy    int32   (N,) – which song each chunk came from
processed/stats.json        dataset statistics
"""
import sys, os, json, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.audio_utils import (
    load_audio, audio_to_melspec, normalize,
    FRAMES_PER_CHUNK, N_MELS,
)

DATA_DIR = Path(ROOT) / "data"
PROCESSED_DIR = Path(ROOT) / "processed"


def chunk_spectrogram(mel_norm: np.ndarray, hop: int) -> list:
    T = mel_norm.shape[1]
    chunks = []
    for start in range(0, T - FRAMES_PER_CHUNK + 1, hop):
        c = mel_norm[:, start : start + FRAMES_PER_CHUNK]
        if c.shape[1] == FRAMES_PER_CHUNK:
            chunks.append(c)
    return chunks


def preprocess_all(
    hop_fraction: float = 0.5,
    log_cb=None,
) -> np.ndarray:
    """Preprocess all MP3s. *log_cb(msg)* receives progress lines (optional)."""
    def _log(msg: str) -> None:
        print(msg)
        if log_cb:
            log_cb(msg)

    PROCESSED_DIR.mkdir(exist_ok=True)

    mp3_files = sorted(DATA_DIR.glob("*.mp3"))
    if not mp3_files:
        raise FileNotFoundError(f"No MP3 files found in {DATA_DIR}")
    _log(f"Found {len(mp3_files)} MP3 files in {DATA_DIR}")

    hop = max(1, int(FRAMES_PER_CHUNK * hop_fraction))
    _log(f"Hop size: {hop} frames  ({hop_fraction:.0%} overlap)")
    all_chunks: list[np.ndarray] = []
    song_indices: list[int] = []

    for song_idx, path in enumerate(mp3_files):
        try:
            audio    = load_audio(str(path))
            mel_db   = audio_to_melspec(audio)
            mel_norm = normalize(mel_db)
            chunks   = chunk_spectrogram(mel_norm, hop)
            all_chunks.extend(chunks)
            song_indices.extend([song_idx] * len(chunks))
            if (song_idx + 1) % 20 == 0 or song_idx == len(mp3_files) - 1:
                _log(f"  {song_idx + 1}/{len(mp3_files)} songs — {len(all_chunks):,} chunks so far")
        except Exception as exc:
            _log(f"  Skipping {path.name}: {exc}")

    chunks_arr = np.array(all_chunks, dtype=np.float32)
    song_arr   = np.array(song_indices, dtype=np.int32)

    np.save(PROCESSED_DIR / "chunks.npy", chunks_arr)
    np.save(PROCESSED_DIR / "song_index.npy", song_arr)

    stats = {
        "n_chunks":         int(len(all_chunks)),
        "n_songs":          int(len(mp3_files)),
        "shape":            list(chunks_arr.shape),
        "mean":             float(chunks_arr.mean()),
        "std":              float(chunks_arr.std()),
        "min":              float(chunks_arr.min()),
        "max":              float(chunks_arr.max()),
        "frames_per_chunk": FRAMES_PER_CHUNK,
        "n_mels":           N_MELS,
        "hop_fraction":     hop_fraction,
        "preprocessed_at":  time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_dir":         str(DATA_DIR),
    }
    with open(PROCESSED_DIR / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Save integrity snapshot so we can detect stale preprocessing later
    try:
        from src.data_integrity import save_snapshot
        save_snapshot(DATA_DIR, PROCESSED_DIR)
    except Exception:
        pass

    size_gb = chunks_arr.nbytes / 1e9
    _log(f"Done. {len(all_chunks):,} chunks from {len(mp3_files)} songs")
    _log(f"Array shape: {chunks_arr.shape}  ({size_gb:.2f} GB on disk)")
    return chunks_arr


if __name__ == "__main__":
    preprocess_all()
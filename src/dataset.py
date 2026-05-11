"""PyTorch datasets for VAE and LSTM-prior training."""
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path

PROCESSED_DIR = Path(ROOT) / "processed"


# ---------------------------------------------------------------------------
# VAE dataset: individual mel-spectrogram chunks
# ---------------------------------------------------------------------------

class MelChunkDataset(Dataset):
    """Single mel-spectrogram chunks for VAE training.
    Uses numpy memory-mapping so multi-GB arrays stay on disk.
    """

    def __init__(self, chunks_path: str, augment: bool = False):
        self.chunks  = np.load(chunks_path, mmap_mode="r")
        self.augment = augment

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> torch.Tensor:
        chunk = np.array(self.chunks[idx], dtype=np.float32)   # copy from mmap

        if self.augment:
            # Gaussian noise
            chunk = chunk + np.random.randn(*chunk.shape).astype(np.float32) * 0.02
            chunk = np.clip(chunk, -1.0, 1.0)

            # Mel-bin pitch shift (roll along frequency axis).
            # Each mel bin ≈ ~0.4 semitone near A4, so ±3 bins ≈ ±1 semitone.
            # This effectively multiplies dataset diversity at zero cost.
            if np.random.rand() < 0.6:
                shift = np.random.choice([-3, -2, -1, 1, 2, 3])
                chunk = np.roll(chunk, shift, axis=0)
                if shift > 0:
                    chunk[:shift, :] = -1.0   # pad vacated bins with silence
                else:
                    chunk[shift:, :] = -1.0

            # Time mask (SpecAugment-lite)
            if np.random.rand() < 0.4:
                t     = np.random.randint(0, max(1, chunk.shape[1] - 10))
                width = np.random.randint(5, 25)
                chunk[:, t : t + width] = -1.0
            # Frequency mask
            if np.random.rand() < 0.3:
                f     = np.random.randint(0, max(1, chunk.shape[0] - 8))
                width = np.random.randint(4, 16)
                chunk[f : f + width, :] = -1.0

        return torch.from_numpy(chunk).unsqueeze(0)   # (1, N_MELS, T)


# ---------------------------------------------------------------------------
# LSTM dataset: consecutive latent-code sequences
# ---------------------------------------------------------------------------

class LatentSequenceDataset(Dataset):
    """Sequences of consecutive VAE latent codes for LSTM training."""

    def __init__(self, latents_path: str, song_index_path: str, seq_len: int = 16):
        self.latents  = np.load(latents_path,    mmap_mode="r")   # (N, latent_dim)
        self.song_idx = np.load(song_index_path)                   # (N,)
        self.seq_len  = seq_len
        self.sequences = self._build_sequences()

    def _build_sequences(self) -> list[tuple[int, int]]:
        """Collect (start, end) index pairs for valid within-song windows."""
        seqs: list[tuple[int, int]] = []
        n = len(self.song_idx)
        i = 0
        while i < n:
            song = self.song_idx[i]
            j = i
            while j < n and self.song_idx[j] == song:
                j += 1
            song_len = j - i
            if song_len >= self.seq_len + 1:
                for start in range(i, j - self.seq_len):
                    seqs.append((start, start + self.seq_len + 1))
            i = j
        return seqs

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start, end = self.sequences[idx]
        seq = np.array(self.latents[start:end], dtype=np.float32)
        x = torch.from_numpy(seq[:-1])   # (seq_len, latent_dim)  input
        y = torch.from_numpy(seq[1:])    # (seq_len, latent_dim)  target
        return x, y


# ---------------------------------------------------------------------------
# DataLoader factories
# ---------------------------------------------------------------------------

def _split_indices(n: int, val_fraction: float) -> tuple[list[int], list[int]]:
    """Reproducible train/val split by index."""
    rng = np.random.default_rng(42)
    idx = rng.permutation(n).tolist()
    val_size  = max(1, int(n * val_fraction))
    return idx[val_size:], idx[:val_size]


def get_vae_loaders(
    chunks_path: str,
    batch_size: int = 32,
    val_fraction: float = 0.05,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader]:
    train_ds = MelChunkDataset(chunks_path, augment=True)
    val_ds   = MelChunkDataset(chunks_path, augment=False)

    train_idx, val_idx = _split_indices(len(train_ds), val_fraction)

    train_loader = DataLoader(
        Subset(train_ds, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        Subset(val_ds, val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader


def get_lstm_loaders(
    latents_path: str,
    song_index_path: str,
    batch_size: int = 128,
    seq_len: int = 16,
    val_fraction: float = 0.05,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader]:
    train_ds = LatentSequenceDataset(latents_path, song_index_path, seq_len=seq_len)
    val_ds   = LatentSequenceDataset(latents_path, song_index_path, seq_len=seq_len)

    train_idx, val_idx = _split_indices(len(train_ds), val_fraction)

    train_loader = DataLoader(
        Subset(train_ds, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        Subset(val_ds, val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader

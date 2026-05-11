"""Hardware detection, VRAM monitoring, and smart training parameter recommendations.

The recommend_params() function does real calculations — not lookup tables —
based on actual VRAM budget, dataset size, and desired gradient-step targets.
See the inline comments for the reasoning behind each value.
"""
import subprocess
import os
from dataclasses import dataclass, asdict
from typing import Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GPUInfo:
    index: int
    name: str
    vram_total_mb: int
    vram_used_mb: int
    vram_free_mb: int
    temperature_c: int
    utilization_pct: int
    power_draw_w: float
    power_limit_w: float

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def vram_pct(self) -> float:
        return 100.0 * self.vram_used_mb / max(1, self.vram_total_mb)


@dataclass
class CPUInfo:
    model: str
    cores_logical: int
    ram_total_gb: float
    ram_available_gb: float

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def ram_pct(self) -> float:
        used = self.ram_total_gb - self.ram_available_gb
        return 100.0 * used / max(0.001, self.ram_total_gb)


# ---------------------------------------------------------------------------
# Hardware queries
# ---------------------------------------------------------------------------

def query_gpus() -> list[GPUInfo]:
    """Query nvidia-smi CSV output for all GPU stats. Returns [] on failure."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free,"
                "temperature.gpu,utilization.gpu,power.draw,power.limit",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        gpus = []
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 9:
                continue
            try:
                gpus.append(GPUInfo(
                    index=int(parts[0]),
                    name=parts[1],
                    vram_total_mb=int(parts[2]),
                    vram_used_mb=int(parts[3]),
                    vram_free_mb=int(parts[4]),
                    temperature_c=int(parts[5]),
                    utilization_pct=int(parts[6].rstrip(" %") or 0),
                    power_draw_w=float(parts[7]),
                    power_limit_w=float(parts[8]),
                ))
            except (ValueError, IndexError):
                pass
        return gpus
    except Exception:
        return []


def query_cpu() -> CPUInfo:
    """Read CPU model from /proc/cpuinfo and RAM from /proc/meminfo."""
    model  = "Unknown CPU"
    cores  = os.cpu_count() or 1

    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass

    ram_total_gb = ram_available_gb = 0.0
    try:
        meminfo: dict[str, int] = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1)
                meminfo[k.strip()] = int(v.split()[0])
        ram_total_gb     = meminfo.get("MemTotal",     0) / 1_000_000
        ram_available_gb = meminfo.get("MemAvailable", 0) / 1_000_000
    except (OSError, ValueError):
        pass

    return CPUInfo(
        model=model,
        cores_logical=cores,
        ram_total_gb=round(ram_total_gb, 1),
        ram_available_gb=round(ram_available_gb, 1),
    )


# ---------------------------------------------------------------------------
# Internal VRAM budget helpers
# ---------------------------------------------------------------------------

# Empirically measured on this codebase:
#   MusicVAE(latent_dim=256), batch=128 → torch.cuda.memory_allocated() ≈ 300 MB
#   Model state (params + grads + Adam m/v) = 17.1M × 4 bytes × 4 = 274 MB
#   → Per-sample activation cost ≈ (300 - 274) / 128 = 0.20 MB
#
# This is dominated by the fixed-size conv feature maps, so it barely
# changes with latent_dim (the FC layers are cheap vs activations).
_PER_SAMPLE_ACT_MB = 0.20   # empirical, conservative

# CUDA runtime + cuDNN workspace + PyTorch internal buffers.
# This is NOT reported by torch.cuda.memory_allocated().
_CUDA_OVERHEAD_MB  = 350


def _vae_state_mb(latent_dim: int) -> float:
    """Model params + gradients + Adam m/v for MusicVAE in MB."""
    flat      = 20_480                           # 256 ch × 8 × 10 after 4 conv-stride-2 blocks
    fc_params = 3 * flat * latent_dim            # enc_mu, enc_logvar, dec_fc
    conv_par  = 1_500_000                        # all conv layers combined (roughly constant)
    total     = fc_params + conv_par
    return total * 4 * 4 / 1e6                   # float32 × (params + grads + Adam m + Adam v)


def _lstm_state_mb(latent_dim: int, hidden: int = 512, n_layers: int = 2) -> float:
    """Model params + Adam state for LatentLSTM in MB."""
    gate_par  = 4 * (latent_dim * hidden + hidden * hidden + hidden)
    lstm_par  = gate_par * n_layers
    proj_par  = hidden * latent_dim
    return (lstm_par + proj_par) * 4 * 4 / 1e6


def _max_vae_batch(free_mb: int, latent_dim: int) -> int:
    """Maximum VAE batch that fits in VRAM with a 1.5× safety margin."""
    budget = free_mb - _CUDA_OVERHEAD_MB - _vae_state_mb(latent_dim)
    if budget <= 0:
        return 8
    # 1.5× safety: never use more than 2/3 of the remaining budget for activations
    return max(8, int(budget / (_PER_SAMPLE_ACT_MB * 1.5)))


# ---------------------------------------------------------------------------
# Core recommendation engine
# ---------------------------------------------------------------------------

def recommend_params(
    gpu: Optional[GPUInfo] = None,
    dataset_stats: Optional[dict] = None,
) -> dict:
    """Return a fully-explained dict of recommended training hyperparameters.

    Parameters
    ----------
    gpu : GPUInfo | None
        Pass an already-queried GPU, or None to auto-detect.
    dataset_stats : dict | None
        Contents of processed/stats.json, or None if preprocessing hasn't run.
        Keys used: n_chunks, n_songs.

    Every recommended value has an entry in ``rationale`` explaining *why*.
    All values are conservative — they will not OOM.
    """
    cpu      = query_cpu()
    warnings = []
    rationale = {}

    # ---- Dataset size -------------------------------------------------------
    n_chunks = dataset_stats.get("n_chunks", 0) if dataset_stats else 0
    n_songs  = dataset_stats.get("n_songs",  0) if dataset_stats else 0

    if n_chunks == 0:
        # Dataset not yet preprocessed — give guidance based on file count only
        n_chunks_est = 0  # unknown
    else:
        n_chunks_est = n_chunks

    n_train = max(1, int(n_chunks_est * 0.95))   # 95% train split

    # ---- Detect GPU ---------------------------------------------------------
    if gpu is None:
        detected = query_gpus()
        gpu = detected[0] if detected else None

    if gpu is None:
        return _cpu_only_params(cpu, n_chunks_est, n_train, warnings)

    free_mb = gpu.vram_free_mb

    # ---- Latent dim ---------------------------------------------------------
    # Higher latent_dim → richer latent space but the FC layers grow as O(latent_dim).
    # At 8GB VRAM with 6.5GB free the model state stays under 350MB for any dim ≤ 256.
    # Benefit diminishes beyond 256 for 11k-chunk datasets (model capacity >> data).
    latent_dim = 256
    rationale["latent_dim"] = (
        f"256 — the FC layers are the main memory cost (O(latent_dim)), "
        f"but at {free_mb}MB free VRAM the state is only {_vae_state_mb(256):.0f}MB. "
        "Larger latent spaces improve reconstruction; diminishing returns beyond 256 "
        "for this dataset size."
    )

    # ---- VAE batch size -----------------------------------------------------
    # Goal: large enough for good GPU occupancy but not so large that we get
    # fewer than ~30 gradient updates per epoch (hurts convergence quality).
    #
    # VRAM budget → theoretical max batch, then we cap at two things:
    #   1. GPU-occupancy floor: ≥32 (too small = GPU starves)
    #   2. Quality cap: n_train // 30  (ensure ≥30 batches/epoch for gradient diversity)
    #   3. Hard cap: 512 (very large batches hurt generalisation on small datasets)
    #
    # Between batch=256 and batch=512:
    #   batch=256 → n_train/256 ≈ 42 gradient steps/epoch → better for small dataset
    #   batch=512 → 21 steps/epoch → fewer updates, can underfit
    # → prefer 256 for this dataset size.

    import math as _math
    vram_max     = _max_vae_batch(free_mb, latent_dim)
    quality_cap  = min(512, max(32, n_train // 30)) if n_train > 0 else 256
    occupancy_fl = 64   # below this the RTX 3070 is poorly utilised for conv ops
    # Snap to power-of-2 for clean GPU memory alignment
    raw_batch    = max(occupancy_fl, min(quality_cap, vram_max))
    vae_batch    = max(occupancy_fl, 2 ** int(_math.log2(raw_batch)))

    rationale["vae_batch_size"] = (
        f"{vae_batch} — VRAM allows up to {vram_max} samples/batch "
        f"({free_mb}MB free − {_CUDA_OVERHEAD_MB}MB CUDA overhead − "
        f"{_vae_state_mb(latent_dim):.0f}MB model state = "
        f"{free_mb - _CUDA_OVERHEAD_MB - _vae_state_mb(latent_dim):.0f}MB ÷ "
        f"{_PER_SAMPLE_ACT_MB:.2f}MB/sample × 0.67 safety factor). "
        f"Quality cap: {quality_cap} (= n_train ÷ 30 to keep ≥30 gradient steps/epoch). "
        "Smaller batches give noisier but more diverse gradients — better for 11k-sample datasets."
    )

    # ---- VAE epochs ----------------------------------------------------------
    # Target: ~150 000 gradient steps (rule of thumb for small-to-medium VAEs).
    # steps = epochs × batches_per_epoch = epochs × (n_train / batch)
    # epochs = target_steps / batches_per_epoch
    # Minimum 200 (the 100-epoch model clearly hasn't converged: val went 0.049→0.013,
    # extrapolation suggests ~0.005 at 500 epochs).
    # Cap at 1000 (after that, further improvement is marginal for this arch/dataset).

    batches_per_epoch_vae = max(1, n_train // vae_batch) if n_train else 40
    target_steps_vae      = 150_000
    vae_epochs            = max(200, min(1000, target_steps_vae // batches_per_epoch_vae))

    rationale["vae_epochs"] = (
        f"{vae_epochs} — targeting {target_steps_vae:,} gradient steps "
        f"({batches_per_epoch_vae} batches/epoch). "
        "The 100-epoch run went from val=0.049 → 0.013; power-law extrapolation "
        "suggests ~0.005 at 500 epochs — a meaningful 2.5× improvement. "
        "More epochs are free on GPU (no data cost)."
    )

    # ---- KL annealing weight ------------------------------------------------
    # Posterior collapse risk is higher on small datasets: the decoder learns to
    # ignore z and the encoder collapses to the prior.
    # Lower kl_max = weaker regularisation = richer latents but possible overfit.
    # 2e-4 is safer than 5e-4 for 11k samples; ramp covers 40% of epochs (not 30%)
    # to give the decoder more time to use latents before KL pressure kicks in.

    if n_chunks_est < 20_000:
        kl_max = 2e-4
        rationale["kl_max"] = (
            "2e-4 — dataset is small (<20k chunks); lower KL weight reduces risk of "
            "posterior collapse (decoder ignoring z). The annealing ramp covers 40% "
            "of training epochs to give the decoder time to use the latent space."
        )
    else:
        kl_max = 5e-4
        rationale["kl_max"] = (
            "5e-4 — default; dataset is large enough that collapse risk is low."
        )

    # ---- LSTM batch size ----------------------------------------------------
    # LSTM sequences are much smaller than mel chunks, and the model is lighter
    # (~4MB state vs 274MB for VAE). Can use a larger batch.
    lstm_state   = _lstm_state_mb(latent_dim)
    lstm_vram_max = max(32, int(
        (free_mb - _CUDA_OVERHEAD_MB - lstm_state) / (0.05)  # seq activations are tiny
    ))
    lstm_batch   = min(512, max(64, lstm_vram_max // 8))

    rationale["lstm_batch_size"] = (
        f"{lstm_batch} — LSTM state is only {lstm_state:.0f}MB; activations per sequence "
        f"are ~0.05MB. Larger batches are safe and give smoother gradient estimates "
        "for sequence modelling."
    )

    # ---- LSTM seq_len -------------------------------------------------------
    # Each seq covers seq_len × 4 s of music. Longer = more temporal context but
    # requires longer songs and more VRAM.
    # Rule: cover ~50% of the median song length (83 chunks × 50% = 41 → round to 40).
    # Must be ≤ min_song_chunks - 1 to produce at least one valid sequence per song.
    if dataset_stats:
        import numpy as np
        median_chunks_per_song = dataset_stats.get("n_chunks", 11433) // max(1, dataset_stats.get("n_songs", 137))
    else:
        median_chunks_per_song = 83  # known from analysis
    seq_len = max(16, min(64, int(median_chunks_per_song * 0.48)))
    # Round to nearest 8 for clean batch sizes
    seq_len = max(16, (seq_len // 8) * 8)

    rationale["seq_len"] = (
        f"{seq_len} — covers ~{seq_len * 4}s of music context (48% of median song length "
        f"= {median_chunks_per_song} chunks × 4s). Longer sequences give the LSTM better "
        "long-range temporal patterns; shorter = more training sequences available."
    )

    # ---- LSTM epochs --------------------------------------------------------
    # LSTM converges faster than VAE (simpler task: predict next latent code).
    # With seq_len=40, approx n_lstm_seqs ≈ n_songs × (median - seq_len)
    n_lstm_seqs     = max(1, n_songs * max(1, median_chunks_per_song - seq_len)) if n_songs else 5000
    batches_per_epoch_lstm = max(1, int(n_lstm_seqs * 0.95) // lstm_batch)
    target_steps_lstm      = 80_000
    lstm_epochs            = max(100, min(500, target_steps_lstm // batches_per_epoch_lstm))

    rationale["lstm_epochs"] = (
        f"{lstm_epochs} — targeting {target_steps_lstm:,} gradient steps "
        f"(~{n_lstm_seqs:,} sequences, {batches_per_epoch_lstm} batches/epoch)."
    )

    # ---- Learning rates -----------------------------------------------------
    # Cosine schedule decays lr to lr × 0.05 over the full run.
    # For 500 epochs, the schedule gives plenty of fine-grained decay.
    # Slightly lower than default 2e-4 to avoid instability at epoch 1 with large batch.
    lr_vae  = 1e-4 if vae_batch >= 256 else 2e-4
    lr_lstm = 5e-4

    rationale["lr_vae"] = (
        f"{lr_vae:.0e} — {'reduced from 2e-4 because ' if vae_batch >= 256 else 'standard; '}"
        f"batch={vae_batch}. Cosine schedule decays to {lr_vae * 0.05:.0e} by the final epoch."
    )
    rationale["lr_lstm"] = (
        f"{lr_lstm:.0e} — standard for LSTM on latent sequences; "
        "cosine schedule handles the decay."
    )

    # ---- num_workers --------------------------------------------------------
    # Physical cores for Ryzen 5 5600G = 6; leave 2 for PyTorch + system.
    # num_workers > batch_size//32 rarely helps.
    phys_cores  = max(1, cpu.cores_logical // 2)
    num_workers = min(phys_cores - 1, max(2, vae_batch // 64))

    rationale["num_workers"] = (
        f"{num_workers} — {phys_cores} physical cores detected, reserving 1 for "
        "PyTorch main thread + system. More workers than this shows no speedup "
        "for this batch size and fast NVMe/SSD access pattern."
    )

    # ---- Griffin-Lim iterations for generation ------------------------------
    rationale["griffin_lim_iters"] = (
        "128 — 64 produces audible metallic wobble; 128 eliminates most of it; "
        "256 gives cleaner phase but costs 2× time. 128 is the quality/speed sweet spot."
    )

    # ---- Temperature default ------------------------------------------------
    rationale["temperature"] = (
        "0.5 — Lambert's style is contemplative and sparse. temp=1.0 over-samples "
        "the prior; 0.5 keeps generated latents close to the data manifold. "
        "Try 0.3 for more predictable/calm output, 0.8 for more experimental."
    )

    # ---- Thermal warnings ---------------------------------------------------
    if gpu.temperature_c >= 85:
        warnings.append(
            f"GPU temperature is {gpu.temperature_c}°C — dangerously high, check cooling before training."
        )
    elif gpu.temperature_c >= 75:
        warnings.append(
            f"GPU temperature is {gpu.temperature_c}°C — monitor thermals during long training runs."
        )

    return {
        "device":             f"cuda:{gpu.index}",
        "latent_dim":         latent_dim,
        "vae_batch_size":     vae_batch,
        "lstm_batch_size":    lstm_batch,
        "vae_epochs":         vae_epochs,
        "lstm_epochs":        lstm_epochs,
        "lr_vae":             lr_vae,
        "lr_lstm":            lr_lstm,
        "kl_max":             kl_max,
        "kl_ramp_pct":        0.4 if (n_chunks_est < 20_000) else 0.3,
        "seq_len":            seq_len,
        "num_workers":        num_workers,
        "checkpoint_every":   50,
        "griffin_lim_iters":  128,
        "temperature":        0.5,
        "cpu_offload":        False,
        "warnings":           warnings,
        "rationale":          rationale,
    }


def _cpu_only_params(cpu, n_chunks, n_train, warnings) -> dict:
    """Fallback params for CPU-only training."""
    warnings.append("No NVIDIA GPU detected — CPU training will be very slow (hours per epoch).")
    return {
        "device":           "cpu",
        "latent_dim":       128,
        "vae_batch_size":   16,
        "lstm_batch_size":  64,
        "vae_epochs":       100,
        "lstm_epochs":      50,
        "lr_vae":           2e-4,
        "lr_lstm":          1e-3,
        "kl_max":           2e-4,
        "seq_len":          16,
        "num_workers":      max(1, cpu.cores_logical // 4),
        "griffin_lim_iters": 64,
        "temperature":      0.5,
        "cpu_offload":      False,
        "warnings":         warnings,
        "rationale":        {"device": "CPU fallback: use minimal batch sizes to avoid RAM pressure."},
    }


# ---------------------------------------------------------------------------
# Preprocessing recommendation
# ---------------------------------------------------------------------------

def recommend_preprocess(
    n_mp3_files: Optional[int] = None,
    current_n_chunks: Optional[int] = None,
) -> dict:
    """Recommend preprocessing settings based on dataset size.

    The single most impactful free improvement for a small dataset is reducing
    hop_fraction: more overlap = more training chunks from the same audio.
    """
    rationale = {}

    if current_n_chunks is None or current_n_chunks == 0:
        # Not yet preprocessed — recommend based on file count
        if n_mp3_files and n_mp3_files < 100:
            hop = 0.2
            rationale["hop_fraction"] = (
                f"0.2 — small collection ({n_mp3_files} files). "
                "25% overlap gives ~4× more chunks vs no overlap. "
                "Each chunk is still 4s long; adjacent chunks share 80% of content "
                "but the variety in boundary conditions helps the VAE generalize."
            )
        else:
            hop = 0.25
            rationale["hop_fraction"] = (
                "0.25 — standard for medium collections. 75% overlap: "
                "each 4s chunk advances ~1s from the previous."
            )
    else:
        # Already preprocessed — compare to target and recommend
        target_chunks = 30_000
        if current_n_chunks >= target_chunks:
            hop = 0.5
            rationale["hop_fraction"] = (
                f"0.5 — dataset ({current_n_chunks:,} chunks) already above the "
                f"{target_chunks:,} target. Standard overlap is fine; "
                "further reduction gives diminishing returns."
            )
        elif current_n_chunks >= 20_000:
            hop = 0.3
            rationale["hop_fraction"] = (
                f"0.3 — current {current_n_chunks:,} chunks is decent but below the "
                f"{target_chunks:,} quality target. Reducing to 0.3 will give "
                f"~{int(current_n_chunks * 0.5/0.3):,} chunks from the same audio."
            )
        else:
            hop = 0.2
            estimated = int(current_n_chunks * 0.5 / 0.2)
            rationale["hop_fraction"] = (
                f"0.2 — current {current_n_chunks:,} chunks is below the "
                f"{target_chunks:,} quality target. Reducing overlap to 0.2 will give "
                f"~{estimated:,} chunks (2.5× more) from the same audio files — "
                "the biggest free quality improvement available."
            )

    return {"hop_fraction": hop, "rationale": rationale}


# ---------------------------------------------------------------------------
# Data adequacy assessment
# ---------------------------------------------------------------------------

def assess_data(
    n_chunks: int,
    n_songs: int,
    hop_fraction: float = 0.5,
) -> dict:
    """Honest assessment of whether the dataset is adequate for good results.

    Returns a dict with:
      tier       : 'insufficient' | 'marginal' | 'decent' | 'good' | 'excellent'
      summary    : one-line verdict
      detail     : paragraph explanation
      actions    : list of specific recommended actions, ordered by impact
    """
    # Effective chunks already account for pitch augmentation at 60% × ±3 variations
    aug_multiplier   = 1.6   # rough effective diversity from pitch shifts
    effective_chunks = int(n_chunks * aug_multiplier)

    if effective_chunks < 10_000:
        tier = "insufficient"
        summary = "Too little data for coherent generation — expect random noise."
    elif effective_chunks < 20_000:
        tier = "marginal"
        summary = "Barely enough to learn tonal palette; temporal structure will be weak."
    elif effective_chunks < 40_000:
        tier = "decent"
        summary = "VAE can learn Lambert's timbre; LSTM temporal coherence is limited."
    elif effective_chunks < 80_000:
        tier = "good"
        summary = "Good quality within the Griffin-Lim vocoder ceiling."
    else:
        tier = "excellent"
        summary = "Dataset is the bottleneck — switch to a neural vocoder for further gains."

    # Chunk rate at current hop_fraction
    current_hop_s = 4.0 * hop_fraction   # seconds per hop step
    hours_of_audio = n_chunks * current_hop_s / 3600

    # What hop=0.2 would give from the same audio
    chunks_at_02 = int(n_chunks * (hop_fraction / 0.2))
    eff_at_02    = int(chunks_at_02 * aug_multiplier)

    # Songs needed for 30k / 60k / 100k chunk targets (at hop=0.2)
    avg_chunks_per_song_at_02 = (n_chunks / max(1, n_songs)) * (hop_fraction / 0.2)
    songs_for_30k  = max(0, int((30_000 / aug_multiplier - chunks_at_02) / max(1, avg_chunks_per_song_at_02)))
    songs_for_60k  = max(0, int((60_000 / aug_multiplier - chunks_at_02) / max(1, avg_chunks_per_song_at_02)))

    actions = []

    # Action 1: re-preprocess with lower hop (always the highest-ROI free action)
    if hop_fraction > 0.21:
        actions.append({
            "priority": "HIGH",
            "cost": "free — same audio files",
            "action": f"Re-preprocess with hop_fraction=0.2",
            "effect": (
                f"Increases chunks {n_chunks:,} → ~{chunks_at_02:,} "
                f"(effective with augmentation: ~{eff_at_02:,}). "
                "This is the single highest-return action available right now."
            ),
        })

    # Action 2: more epochs
    if tier in ("marginal", "decent"):
        actions.append({
            "priority": "HIGH",
            "cost": "free — just training time",
            "action": "Use 400-500 VAE epochs, 200 LSTM epochs",
            "effect": (
                "The 100-epoch model converged to val=0.013. Extrapolation suggests "
                "~0.005 at 500 epochs — a 2.5× reduction in reconstruction error. "
                "With ~40 batches/epoch this takes a few minutes more on RTX 3070."
            ),
        })

    # Action 3: get more Lambert songs
    if tier in ("insufficient", "marginal", "decent"):
        if songs_for_30k == 0:
            # Already at or past the 30k threshold with hop=0.2
            actions.append({
                "priority": "MEDIUM",
                "cost": "manual — download more albums",
                "action": f"Add ~{max(1, songs_for_60k)} more Lambert songs to reach 60k-chunk tier",
                "effect": (
                    f"At hop=0.2 your existing {n_songs} tracks are sufficient for 'decent' quality "
                    f"(~{chunks_at_02:,} chunks). For 'good' you need ~{max(1, songs_for_60k)} more songs. "
                    "Lambert has 10+ studio albums. Stick to Lambert specifically — "
                    "mixing in other artists (Nils Frahm, Max Richter) adds variety but blurs the style."
                ),
            })
        else:
            actions.append({
                "priority": "MEDIUM",
                "cost": "manual — download more albums",
                "action": f"Add ~{songs_for_30k} more Lambert songs to reach 30k-chunk tier",
                "effect": (
                    f"Lambert has 10+ studio albums; you have {n_songs} tracks. "
                    f"At hop=0.2 you need ~{songs_for_30k} more songs for 'decent' quality "
                    f"(~{songs_for_60k} more for 'good'). "
                    "Stick to Lambert specifically — mixing in other artists blurs the style."
                ),
            })

    # Action 4: pitch-shifted copies at audio level (sox)
    actions.append({
        "priority": "MEDIUM",
        "cost": "automated — sox pitch shift",
        "action": "Generate ±2 semitone transpositions of each MP3 with sox",
        "effect": (
            f"Creates {n_songs * 4} more files (±1, ±2 semitones) from existing data. "
            f"Increases effective dataset to ~{n_chunks * 5:,} raw chunks "
            "(5× current). Higher fidelity than the mel-bin roll augmentation already built in."
            "\n  Example: for f in data/*.mp3; do "
            "sox \"$f\" \"${f%.mp3}_+1.mp3\" pitch 100; done"
        ),
    })

    # Action 5: better vocoder (architectural improvement)
    actions.append({
        "priority": "LOW (architectural change)",
        "cost": "complex — requires separate vocoder training",
        "action": "Replace Griffin-Lim with a neural vocoder (HiFi-GAN)",
        "effect": (
            "Griffin-Lim is the current quality ceiling — even with perfect spectrograms "
            "it produces metallic/phasey artefacts. HiFi-GAN would sound dramatically better "
            "but requires a separate GAN to train on ~1-10h of audio, plus inference integration. "
            "Not worth doing until the VAE itself reconstructs well (val_loss < 0.003)."
        ),
    })

    return {
        "tier":             tier,
        "summary":          summary,
        "n_chunks":         n_chunks,
        "n_songs":          n_songs,
        "effective_chunks": effective_chunks,
        "hours_of_audio":   round(hours_of_audio, 1),
        "hop_fraction":     hop_fraction,
        "actions":          actions,
    }


# ---------------------------------------------------------------------------
# VRAM estimation (for the UI estimate bars)
# ---------------------------------------------------------------------------

def estimate_vram_mb(
    vae_batch_size: int,
    latent_dim: int,
    lstm_batch_size: int = 64,
    seq_len: int = 16,
    n_mels: int = 128,
    n_frames: int = 172,
) -> dict:
    """Return VRAM estimates (MB) for VAE and LSTM training stages."""
    vae_state   = _vae_state_mb(latent_dim)
    vae_act     = vae_batch_size * _PER_SAMPLE_ACT_MB
    vae_total   = vae_state + vae_act + _CUDA_OVERHEAD_MB

    lstm_state  = _lstm_state_mb(latent_dim)
    lstm_seq_mb = lstm_batch_size * seq_len * latent_dim * 4 / 1e6
    lstm_act    = lstm_seq_mb * 5
    lstm_total  = lstm_state + lstm_act + _CUDA_OVERHEAD_MB

    return {
        "vae_mb":  round(vae_total),
        "lstm_mb": round(lstm_total),
        "vae_breakdown": {
            "model_state_mb":   round(vae_state),
            "activations_mb":   round(vae_act, 1),
            "cuda_overhead_mb": _CUDA_OVERHEAD_MB,
        },
        "lstm_breakdown": {
            "model_state_mb":   round(lstm_state),
            "sequences_mb":     round(lstm_seq_mb, 1),
            "activations_mb":   round(lstm_act, 1),
            "cuda_overhead_mb": _CUDA_OVERHEAD_MB,
        },
    }


def vram_warning(estimated_mb: int, free_mb: int, stage: str = "VAE") -> Optional[str]:
    if free_mb <= 0:
        return None
    pct = 100.0 * estimated_mb / free_mb
    if pct >= 100:
        return (
            f"⚠ {stage} estimate ({estimated_mb} MB) EXCEEDS available VRAM ({free_mb} MB). "
            "Reduce batch size or switch to CPU."
        )
    if pct >= 85:
        return (
            f"⚠ {stage} estimate ({estimated_mb} MB) is {pct:.0f}% of free VRAM ({free_mb} MB). "
            "Risk of OOM — consider reducing batch size."
        )
    return None

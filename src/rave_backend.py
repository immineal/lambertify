"""RAVE (Realtime Audio Variational autoEncoder) backend.

Correct CLI usage (all run from ROOT dir with cwd=ROOT):
  rave preprocess --input_path data/ --output_path processed/rave_preprocessed
                  --sampling_rate 44100 --num_signal 131072

  rave train --config v2.gin --db_path processed/rave_preprocessed
             --name lambert --batch 8 --n_signal 131072 --max_steps 500000
             --val_every 10000 --workers 4
  → creates: ROOT/runs/lambert_<gin_hash>/

  rave export --run runs/lambert_<gin_hash>
  → creates: runs/lambert_<gin_hash>/<name>.ts

Reference: https://github.com/acids-icam/RAVE
"""
import glob
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

ROOT          = Path(__file__).parent.parent
DATA_DIR      = ROOT / "data"
RAVE_DATA_DIR = ROOT / "processed" / "rave_preprocessed"
# RAVE always writes runs/ relative to CWD.  We set cwd=ROOT, so:
RAVE_RUNS_DIR = ROOT / "runs"

RAVE_BINARY = Path(sys.executable).parent / "rave"

LogCB    = Optional[Callable[[str], None]]
StepCB   = Optional[Callable[[int], None]]   # called with current step number


# ---------------------------------------------------------------------------
# Installation check
# ---------------------------------------------------------------------------

_install_cache: bool | None = None  # cached after first check


def is_installed() -> bool:
    """Check whether the rave binary is present and importable.

    Result is cached after the first call — the binary doesn't un-install itself
    at runtime, so repeated calls from the polling loop are essentially free.
    The binary-existence check is synchronous and takes ~1 ms; we avoid spawning
    a full Python subprocess on every API call.
    """
    global _install_cache
    if _install_cache is not None:
        return _install_cache
    if not RAVE_BINARY.exists():
        _install_cache = False
        return False
    # Quick check: run `rave --help` (no Python import chain, <0.1 s)
    try:
        r = subprocess.run(
            [str(RAVE_BINARY), "--help"],
            capture_output=True, timeout=5,
        )
        _install_cache = r.returncode == 0
        return _install_cache
    except Exception:
        _install_cache = False
        return False


def invalidate_install_cache() -> None:
    """Call after installing or uninstalling acids-rave."""
    global _install_cache
    _install_cache = None


def install_error() -> str | None:
    """Return the last traceback line if rave is broken, or None if it's fine.

    Only called when is_installed() returns True but something looks wrong.
    Spawns a subprocess — don't call this on every request.
    """
    if not RAVE_BINARY.exists():
        return "rave binary not found — install acids-rave first"
    try:
        r = subprocess.run(
            [sys.executable, "-c", "import rave"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode != 0:
            lines = (r.stderr or r.stdout).strip().splitlines()
            return lines[-1] if lines else "Unknown import error"
        return None
    except Exception as exc:
        return str(exc)


def install(log_cb: LogCB = None) -> bool:
    _log = log_cb or print
    _log("pip install acids-rave …")
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "pip", "install", "acids-rave"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        for line in proc.stdout:
            _log(line.rstrip())
        proc.wait()
        if proc.returncode == 0:
            _log("acids-rave installed.")
            return True
        _log(f"pip failed (exit {proc.returncode})")
        return False
    except Exception as exc:
        _log(f"Install error: {exc}")
        return False


# ---------------------------------------------------------------------------
# Smart defaults
# ---------------------------------------------------------------------------

def recommend_params(gpu=None, n_songs: int = 137, hours_of_audio: float = 6.35) -> dict:
    from src.hardware import query_gpus

    warnings  = []
    rationale = {}

    if gpu is None:
        gpus = query_gpus()
        gpu  = gpus[0] if gpus else None

    sample_rate = 44100
    rationale["sample_rate"] = "44100 Hz — RAVE v2 native; best piano quality"
    rationale["config"]      = "v2 — best for tonal/melodic content (piano = ideal)"

    # n_signal MUST match what was used in preprocessing (stored in metadata.yaml).
    # We preprocessed with --num_signal 131072; using a smaller n_signal at train
    # time causes the MultiScaleSpectralDiscriminator to receive signals too short
    # for its STFT window.  Always use 131072 to stay safe.
    n_signal = 131072
    if gpu and gpu.vram_free_mb >= 5000:
        batch = 8
        rationale["n_signal"]   = "131072 — must match preprocessing (num_signal=131072); ~3 s context"
        rationale["batch_size"] = "8 — smooth gradients, ~2-3 GB VRAM on RTX 3070"
    elif gpu and gpu.vram_free_mb >= 3000:
        batch = 4
        rationale["n_signal"]   = "131072 — must match preprocessing"
        rationale["batch_size"] = "4 — reduced for VRAM"
    else:
        batch = 2
        rationale["n_signal"]   = "131072 — must match preprocessing"
        rationale["batch_size"] = "2 — minimal for VRAM"
        warnings.append("Very low VRAM — batch=2; training will be slow")

    secs_per_step = 0.06   # empirical on RTX 3070, batch=8, n_signal=131072
    steps_quick   = 100_000
    steps_good    = 500_000
    steps_best    = 2_000_000
    rationale["steps"] = (
        f"100k ≈ {steps_quick*secs_per_step/3600:.1f}h — hear if it's working\n"
        f"500k ≈ {steps_good*secs_per_step/3600:.1f}h — recognisable Lambert style\n"
        f"2M   ≈ {steps_best*secs_per_step/3600:.0f}h — full quality"
    )

    workers = min(4, max(1, (os.cpu_count() or 2) // 2))
    rationale["workers"] = f"{workers} — half CPU cores"

    if hours_of_audio < 3:
        warnings.append(f"Only {hours_of_audio:.1f}h audio — 3+ hours gives better results")

    return {
        "config":       "v2",
        "sample_rate":  sample_rate,
        "n_signal":     n_signal,
        "batch_size":   batch,
        "n_steps":      steps_good,
        "steps_quick":  steps_quick,
        "steps_good":   steps_good,
        "steps_best":   steps_best,
        "workers":      workers,
        "warnings":     warnings,
        "rationale":    rationale,
    }


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

# This value is fixed for the current dataset and must match between preprocess and train.
RAVE_NUM_SIGNAL = 131072


def preprocess(
    data_dir: str | None   = None,
    output_dir: str | None = None,
    sample_rate: int       = 44100,
    log_cb: LogCB          = None,
) -> str:
    data_dir   = data_dir   or str(DATA_DIR)
    output_dir = output_dir or str(RAVE_DATA_DIR)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cmd = [
        str(RAVE_BINARY),
        "preprocess",
        "--input_path",    data_dir,
        "--output_path",   output_dir,
        "--sampling_rate", str(sample_rate),
        "--num_signal",    str(RAVE_NUM_SIGNAL),
    ]
    _run_logged(cmd, log_cb, "preprocess", cwd=str(ROOT))

    # Save integrity snapshot so stale checks work
    try:
        from src.data_integrity import save_snapshot
        save_snapshot(data_dir, output_dir)
    except Exception:
        pass

    return output_dir


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

_proc_lock  = threading.Lock()
_train_proc: subprocess.Popen | None = None


def find_run_dir(name: str) -> Path | None:
    """Find the run directory created by RAVE (name + gin hash suffix)."""
    pattern = str(RAVE_RUNS_DIR / f"{name}_*")
    dirs    = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    return Path(dirs[0]) if dirs else None


def train(
    name: str,
    config: str         = "v2",
    n_signal: int       = RAVE_NUM_SIGNAL,
    batch_size: int     = 8,
    n_steps: int        = 500_000,
    workers: int        = 4,
    sample_rate: int    = 44100,
    db_path: str | None = None,
    log_cb:  LogCB      = None,
    step_cb: StepCB     = None,
    stop_event: threading.Event | None = None,
) -> str | None:
    """Run rave train. Returns exported .ts model path on success, None on failure."""
    global _train_proc

    db_path = db_path or str(RAVE_DATA_DIR)

    # Ensure config has .gin extension
    config_arg = config if config.endswith(".gin") else f"{config}.gin"

    # Detect available GPU index; fall back to CPU flag (-1) if none
    import torch as _torch
    if _torch.cuda.is_available():
        gpu_flags = ["--gpu", "0"]
    else:
        gpu_flags = ["--gpu", "-1"]

    # val_every controls how often RAVE saves a checkpoint.
    # With ~1103 steps/epoch, val_every=10000 means ~9 epochs between saves.
    # If training is stopped before the first save the export gets a random model.
    # Scale to ~every 2 epochs, capped at 5000 so short runs still checkpoint.
    steps_per_epoch_est = 1100   # empirical for this dataset at batch=8
    val_every = max(1000, min(5000, steps_per_epoch_est * 2))

    cmd = [
        str(RAVE_BINARY),
        "train",
        "--config",    config_arg,
        "--db_path",   db_path,
        "--name",      name,
        "--n_signal",  str(n_signal),
        "--batch",     str(batch_size),
        "--max_steps", str(n_steps),
        "--val_every", str(val_every),
        "--workers",   str(workers),
        *gpu_flags,
    ]

    if log_cb:
        log_cb(f"[rave train] {' '.join(cmd)}")
        log_cb(f"[rave train] cwd={ROOT}")

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}

    # Binary mode so we can distinguish \r (tqdm in-place update) from \n (real log line).
    # text=True + universal_newlines converts \r → \n, flooding the log with tqdm frames.
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=False, bufsize=0, cwd=str(ROOT), env=env,
    )
    with _proc_lock:
        _train_proc = proc

    # Step tracking: tqdm shows per-epoch steps (e.g. "97/1103 [...]").
    # We reconstruct the global step: epoch * steps_per_epoch + current_in_epoch.
    _current_epoch    = 0
    _steps_per_epoch  = 0     # learned from first tqdm denominator
    _global_step      = 0
    _buf              = b""
    _last_prog_log    = 0.0   # throttle progress-bar log lines
    _PROG_INTERVAL    = 15.0  # seconds between logged progress updates

    # Compiled regex that matches every category of RAVE startup noise.
    # Tested against real output — see commit message for the full list.
    _NOISE_RE = re.compile(
        r'^\s*\)\s*$'                       # closing bracket on its own line
        r'|^\s*\('                          # opening bracket (layer descriptions)
        r'|^\s*\d+\s*\|'                    # "0 | pqmf | ..." table rows
        r'|^\s*\|'                          # "| Name | Type |" header row
        r'|^\s*-{3,}'                       # "---" separator lines
        r'|^\s*\d[\d.]*\s+[MK]\b'          # "58.7 M  Trainable params"
        r'|^\s*\d+\s+[MK]\b'               # "0  M  Non-trainable"
        r'|\bTPU available\b'
        r'|\bIPU available\b'
        r'|\bHPU available\b'
        r'|\bLOCAL_RANK\b'
        r'|\bUserWarning\b'
        r'|warning_cache\.warn'
        r'|\bWeightNorm\b'
        r'|\bFutureWarning\b'
        r'|torch\.nn\.utils\.weight_norm'
        r'|kernel_size=|stride=|negative_slope='
        r'|ModuleList|CachedSequential'
        r'|Trainable params|Non-trainable params|Total params|params size'
        r'|^/.*\.py:\d+:',                  # "/path/to/file.py:232: UserWarning..."
    )

    def _is_noise(line: str) -> bool:
        return bool(_NOISE_RE.search(line))

    def _handle_line(raw_line: bytes, is_progress: bool) -> None:
        nonlocal _current_epoch, _steps_per_epoch, _global_step, _last_prog_log

        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line:
            return

        if _is_noise(line):
            return

        # Parse "Epoch X:" to track which epoch we're in
        m_ep = re.search(r"Epoch\s+(\d+)\s*:", line)
        if m_ep:
            _current_epoch = int(m_ep.group(1))

        # Parse "step_in_epoch / steps_per_epoch [" from tqdm bar
        m_steps = re.search(r"[\|\s](\d+)/(\d+)\s*\[", line)
        if m_steps:
            in_epoch        = int(m_steps.group(1))
            _steps_per_epoch = int(m_steps.group(2))
            _global_step    = _current_epoch * _steps_per_epoch + in_epoch
            if step_cb:
                step_cb(_global_step)

        if is_progress:
            now = time.time()
            if now - _last_prog_log < _PROG_INTERVAL:
                return
            _last_prog_log = now
            # Emit a tidy one-liner: strip the block-char bar, keep numbers
            clean = re.sub(r"[█▏▎▍▌▋▊▉│]{1,}\s*\|", "", line)
            clean = re.sub(r"\s{2,}", " ", clean).strip()
            if log_cb:
                log_cb(f"[step {_global_step:,}] {clean}")
        else:
            if log_cb:
                log_cb(line)

    for chunk in iter(lambda: proc.stdout.read(256), b""):
        _buf += chunk
        # Process all complete lines in the buffer
        while True:
            ni = _buf.find(b"\n")
            ri = _buf.find(b"\r")
            if ni == -1 and ri == -1:
                break
            if ni == -1 or (ri != -1 and ri < ni):
                # \r comes first → tqdm in-place update
                _handle_line(_buf[:ri], is_progress=True)
                _buf = _buf[ri + 1:]
                # If immediately followed by \n (Windows \r\n), skip the \n
                if _buf.startswith(b"\n"):
                    _buf = _buf[1:]
            else:
                # \n comes first → real log line
                _handle_line(_buf[:ni], is_progress=False)
                _buf = _buf[ni + 1:]

        if stop_event and stop_event.is_set():
            proc.terminate()
            if log_cb:
                log_cb("[RAVE] SIGTERM sent — waiting for checkpoint flush…")
            # Give pytorch-lightning time to write last.ckpt before we export
            time.sleep(3)
            break

    # Flush any remaining partial line
    if _buf.strip():
        _handle_line(_buf, is_progress=False)

    proc.wait()
    with _proc_lock:
        _train_proc = None

    success = proc.returncode in (0, -15)   # 0=ok, -15=SIGTERM (user stop)
    if not success:
        if log_cb:
            log_cb(f"[rave train] exited with code {proc.returncode}")
        return None

    return _export(name, log_cb)


def stop_training() -> None:
    with _proc_lock:
        if _train_proc:
            _train_proc.terminate()


def _export(name: str, log_cb: LogCB = None) -> str | None:
    run_dir = find_run_dir(name)
    if not run_dir:
        if log_cb:
            log_cb(f"[rave export] ERROR: no run dir found matching runs/{name}_*")
        return None

    if log_cb:
        log_cb(f"[rave export] exporting {run_dir} …")

    cmd = [str(RAVE_BINARY), "export", "--run", str(run_dir)]
    try:
        _run_logged(cmd, log_cb, "rave export", cwd=str(ROOT))
    except Exception as exc:
        if log_cb:
            log_cb(f"[rave export] failed: {exc}")
        return None

    # Find exported .ts file anywhere under run_dir or ROOT
    for search_root in (run_dir, ROOT):
        matches = sorted(search_root.rglob(f"{name}*.ts"), key=os.path.getmtime, reverse=True)
        if matches:
            if log_cb:
                log_cb(f"[rave export] → {matches[0]}")
            return str(matches[0])

    if log_cb:
        log_cb("[rave export] WARNING: no .ts file found after export")
    return None


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate(
    model_path: str,
    duration_s: float  = 20.0,
    temperature: float = 0.5,
    output_path: str | None = None,
    device_str: str    = "cuda",
) -> str:
    """Generate audio from an exported RAVE .ts model. Returns WAV path."""
    import torch, numpy as np, soundfile as sf

    output_path = output_path or str(ROOT / "outputs" / f"rave_{int(time.time())}.wav")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    model  = torch.jit.load(model_path, map_location=device)
    model.eval()

    try:
        sr = int(model.sr)
    except Exception:
        sr = 44100

    probe_len = 8192
    try:
        with torch.no_grad():
            z_probe    = model.encode(torch.zeros(1, 1, probe_len, device=device))
        latent_dim = z_probe.shape[1]
        ratio      = max(1, probe_len // max(1, z_probe.shape[2]))
    except Exception:
        latent_dim, ratio = 1, 2048

    z = torch.randn(1, latent_dim, max(1, int(duration_s * sr / ratio)),
                    device=device) * temperature

    with torch.no_grad():
        y = model.decode(z)

    audio = y.squeeze().cpu().numpy()
    if audio.ndim > 1:
        audio = audio[0]
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * 0.95

    sf.write(output_path, audio.astype(np.float32), sr)
    return output_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_logged(cmd: list, log_cb: LogCB, label: str, cwd: str | None = None) -> None:
    env  = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, cwd=cwd, env=env,
    )
    for raw in proc.stdout:
        line = raw.rstrip()
        if line and log_cb:
            log_cb(line)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"{label} exited with code {proc.returncode}")


def preprocess_ready(data_dir: str | None = None) -> bool:
    """True if RAVE preprocessing completed AND data hasn't changed since."""
    if not (RAVE_DATA_DIR / "metadata.yaml").exists():
        return False
    # Also fail if the data directory has changed since preprocessing
    from src.data_integrity import check_stale
    dd = data_dir or str(DATA_DIR)
    result = check_stale(dd, RAVE_DATA_DIR)
    return not result["stale"]


def preprocess_info() -> dict:
    """Return RAVE preprocessing metadata if available, else empty dict."""
    meta_path = RAVE_DATA_DIR / "metadata.yaml"
    if not meta_path.exists():
        return {}
    try:
        import yaml
        with open(meta_path) as f:
            d = yaml.safe_load(f) or {}
        d["n_hours"] = round(d.get("n_seconds", 0) / 3600, 2)
        # Add snapshot info if available
        from src.data_integrity import load_snapshot
        snap = load_snapshot(RAVE_DATA_DIR)
        if snap:
            d["preprocessed_at"] = snap.get("preprocessed_at")
            d["n_files"]         = snap.get("n_files")
        return d
    except Exception:
        return {}


def estimate_vram_mb(batch_size: int = 8, n_signal: int = 131072) -> dict:
    """Rough RAVE v2 VRAM estimate.

    RAVE v2 with capacity=64:
      - Model (encoder + decoder + discriminators): ~800 MB
      - Per-sample activations: ~3 MB for n_signal=131072
      - PyTorch overhead: ~300 MB
    """
    model_mb   = 800
    per_sample = 3.0   # MB per audio sample in the batch
    act_mb     = batch_size * per_sample
    overhead   = 300
    total      = model_mb + act_mb + overhead
    return {
        "total_mb":  round(total),
        "model_mb":  model_mb,
        "act_mb":    round(act_mb),
        "overhead":  overhead,
        "note":      f"RAVE v2, batch={batch_size}, n_signal={n_signal}",
    }


def training_ready() -> bool:
    """True if at least one RAVE run directory exists."""
    return RAVE_RUNS_DIR.exists() and any(RAVE_RUNS_DIR.iterdir())

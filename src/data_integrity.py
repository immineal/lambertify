"""Data integrity checks: detect when source data has changed since preprocessing.

A lightweight snapshot (file count + newest mtime + total size) is stored
alongside the preprocessed output.  On each page load we compare the current
data directory against the snapshot and warn if they diverge.
"""
import json
import os
import time
from pathlib import Path
from typing import Optional


def scan_dir(data_dir: str | Path, extensions: tuple = (".mp3", ".wav", ".flac", ".ogg", ".aac", ".opus")) -> dict:
    """Return a lightweight fingerprint of an audio directory."""
    p   = Path(data_dir)
    if not p.exists():
        return {"exists": False, "n_files": 0, "total_bytes": 0, "newest_mtime": 0}
    files = [f for f in p.iterdir() if f.suffix.lower() in extensions]
    if not files:
        return {"exists": True, "n_files": 0, "total_bytes": 0, "newest_mtime": 0}
    return {
        "exists":       True,
        "n_files":      len(files),
        "total_bytes":  sum(f.stat().st_size for f in files),
        "newest_mtime": max(f.stat().st_mtime for f in files),
        "newest_mtime_str": time.strftime(
            "%Y-%m-%d %H:%M", time.localtime(max(f.stat().st_mtime for f in files))
        ),
    }


def snapshot_path(processed_dir: str | Path) -> Path:
    return Path(processed_dir) / "data_snapshot.json"


def save_snapshot(data_dir: str | Path, processed_dir: str | Path) -> dict:
    """Save a snapshot of data_dir into processed_dir/data_snapshot.json."""
    snap = {
        **scan_dir(data_dir),
        "data_dir":       str(data_dir),
        "preprocessed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "preprocessed_ts": time.time(),
    }
    sp = snapshot_path(processed_dir)
    sp.parent.mkdir(parents=True, exist_ok=True)
    with open(sp, "w") as f:
        json.dump(snap, f, indent=2)
    return snap


def load_snapshot(processed_dir: str | Path) -> Optional[dict]:
    sp = snapshot_path(processed_dir)
    if not sp.exists():
        return None
    try:
        with open(sp) as f:
            return json.load(f)
    except Exception:
        return None


def check_stale(data_dir: str | Path, processed_dir: str | Path) -> dict:
    """Compare current data_dir against the stored snapshot.

    Returns a dict with:
        stale        : bool — True if a re-preprocess is recommended
        reason       : str  — human-readable reason (empty if not stale)
        current      : dict — current scan result
        snapshot     : dict — what was recorded at preprocess time (or None)
    """
    current  = scan_dir(data_dir)
    snapshot = load_snapshot(processed_dir)

    if snapshot is None:
        return {
            "stale": True, "current": current, "snapshot": None,
            "reason": "No preprocessing snapshot found — run preprocessing first.",
        }

    if not current["exists"] or current["n_files"] == 0:
        return {
            "stale": True, "current": current, "snapshot": snapshot,
            "reason": f"Data directory is empty or missing: {data_dir}",
        }

    reasons = []
    if current["n_files"] != snapshot.get("n_files", 0):
        diff = current["n_files"] - snapshot.get("n_files", 0)
        reasons.append(
            f"File count changed: {snapshot['n_files']} → {current['n_files']} "
            f"({'+ ' if diff > 0 else ''}{diff} files)"
        )

    prep_ts = snapshot.get("preprocessed_ts", 0)
    if current["newest_mtime"] > prep_ts:
        reasons.append(
            f"New or modified files since last preprocessing "
            f"(newest: {current['newest_mtime_str']}, "
            f"preprocessed: {snapshot.get('preprocessed_at', '?')})"
        )

    return {
        "stale":    bool(reasons),
        "reason":   " · ".join(reasons),
        "current":  current,
        "snapshot": snapshot,
    }

"""Persistent user configuration stored in config.json at the project root.

Saves and restores settings that the user has changed manually: data folder,
RAVE training params, last-used VAE params, etc.  All values have sensible
defaults so the file is optional.
"""
import json
import time
from pathlib import Path

ROOT        = Path(__file__).parent.parent
CONFIG_PATH = ROOT / "config.json"

_DEFAULTS: dict = {
    "data_dir": str(ROOT / "data"),
    "rave": {
        "name":        "lambert",
        "config":      "v2",
        "batch_size":  8,
        "n_steps":     500_000,
        "workers":     4,
        "sample_rate": 44100,
    },
    "vae": {},      # filled by smart-defaults, stored here so UI restores them
    "updated_at":   None,
}


def load() -> dict:
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH) as f:
                stored = json.load(f)
            # Deep-merge: stored values override defaults, missing keys use defaults
            merged = dict(_DEFAULTS)
            for k, v in stored.items():
                if isinstance(v, dict) and isinstance(merged.get(k), dict):
                    merged[k] = {**merged[k], **v}
                else:
                    merged[k] = v
            return merged
        except Exception:
            pass
    return dict(_DEFAULTS)


def save(data: dict) -> None:
    out = dict(data)
    out["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(CONFIG_PATH, "w") as f:
        json.dump(out, f, indent=2)


def get(key: str, default=None):
    return load().get(key, default)


def set_key(key: str, value) -> dict:
    d = load()
    d[key] = value
    save(d)
    return d


def update_section(section: str, **kwargs) -> dict:
    """Merge kwargs into config[section] and persist."""
    d = load()
    d.setdefault(section, {})
    d[section].update(kwargs)
    save(d)
    return d

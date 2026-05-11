"""Model registry and in-memory cache.

Keeps track of every trained checkpoint (VAE+LSTM and RAVE), lets the user
switch which model is "active" for generation, and caches loaded models in
GPU/CPU memory so they don't reload on every generation call.

Registry lives at: checkpoints/models.json
Per-run dirs live at: checkpoints/runs/<run_id>/
RAVE models live at:  checkpoints/rave/<name>/
"""
import json
import shutil
import time
import os
from pathlib import Path
from typing import Optional

ROOT           = Path(__file__).parent.parent
CHECKPOINT_DIR = ROOT / "checkpoints"
RUNS_DIR       = CHECKPOINT_DIR / "runs"
RAVE_DIR       = CHECKPOINT_DIR / "rave"
REGISTRY_PATH  = CHECKPOINT_DIR / "models.json"

for _d in (CHECKPOINT_DIR, RUNS_DIR, RAVE_DIR):
    _d.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Registry  (persistent JSON)
# ---------------------------------------------------------------------------

class ModelRegistry:
    """Thread-safe(ish) persistent model registry."""

    def __init__(self):
        self._path = REGISTRY_PATH
        self._data = self._load()
        self._migrate_legacy()

    def _load(self) -> dict:
        if self._path.exists():
            try:
                with open(self._path) as f:
                    return json.load(f)
            except Exception:
                pass
        return {"models": [], "active_model": None}

    def _save(self) -> None:
        with open(self._path, "w") as f:
            json.dump(self._data, f, indent=2)

    def _migrate_legacy(self) -> None:
        """Import pre-registry checkpoints sitting in checkpoints/ root."""
        known_ids = {m["id"] for m in self._data["models"]}
        changed   = False

        for pt in CHECKPOINT_DIR.glob("vae_*.pt"):
            model_id = f"legacy_{pt.stem}"
            if model_id in known_ids:
                continue
            try:
                import torch
                ckpt = torch.load(pt, map_location="cpu", weights_only=False)
                lstm_pt = CHECKPOINT_DIR / "lstm_best.pt"
                entry = {
                    "id":          model_id,
                    "name":        f"Legacy — {pt.name}",
                    "backend":     "vae_lstm",
                    "created":     time.strftime("%Y-%m-%d %H:%M:%S",
                                                  time.localtime(pt.stat().st_mtime)),
                    "vae_path":    str(pt),
                    "lstm_path":   str(lstm_pt) if lstm_pt.exists() else None,
                    "config":      ckpt.get("config", {}),
                    "vae_epoch":   ckpt.get("epoch"),
                    "vae_val_loss": ckpt.get("val_loss"),
                    "checkpoints": [],
                }
                self._data["models"].append(entry)
                known_ids.add(model_id)
                if not self._data["active_model"]:
                    self._data["active_model"] = model_id
                changed = True
            except Exception:
                pass

        if changed:
            self._save()

    # ---- CRUD ---------------------------------------------------------------

    def register_vae_lstm(
        self,
        run_id: str,
        vae_path: str,
        lstm_path: Optional[str],
        config: dict,
        vae_epoch: int,
        vae_val_loss: float,
        name: str | None = None,
        make_active: bool = True,
    ) -> dict:
        entry = {
            "id":          run_id,
            "name":        name or f"VAE+LSTM  {time.strftime('%Y-%m-%d %H:%M')}",
            "backend":     "vae_lstm",
            "created":     time.strftime("%Y-%m-%d %H:%M:%S"),
            "vae_path":    vae_path,
            "lstm_path":   lstm_path,
            "config":      config,
            "vae_epoch":   vae_epoch,
            "vae_val_loss": vae_val_loss,
            "checkpoints": [],
        }
        # update if run_id already exists (repeated best saves during training)
        for i, m in enumerate(self._data["models"]):
            if m["id"] == run_id:
                self._data["models"][i] = {**m, **entry, "checkpoints": m.get("checkpoints", [])}
                if make_active:
                    self._data["active_model"] = run_id
                self._save()
                return self._data["models"][i]

        self._data["models"].append(entry)
        if make_active:
            self._data["active_model"] = run_id
        self._save()
        return entry

    def add_checkpoint(self, run_id: str, epoch: int, path: str) -> None:
        for m in self._data["models"]:
            if m["id"] == run_id:
                ckpts = m.setdefault("checkpoints", [])
                if not any(c["epoch"] == epoch for c in ckpts):
                    ckpts.append({"epoch": epoch, "path": path})
                self._save()
                return

    def register_rave(
        self,
        model_id: str,
        model_path: str,
        config: dict,
        steps: int = 0,
        name: str | None = None,
        make_active: bool = True,
    ) -> dict:
        entry = {
            "id":       model_id,
            "name":     name or f"RAVE  {time.strftime('%Y-%m-%d %H:%M')}",
            "backend":  "rave",
            "created":  time.strftime("%Y-%m-%d %H:%M:%S"),
            "rave_path": model_path,
            "config":   config,
            "steps":    steps,
        }
        for i, m in enumerate(self._data["models"]):
            if m["id"] == model_id:
                self._data["models"][i] = {**m, **entry}
                if make_active:
                    self._data["active_model"] = model_id
                self._save()
                return self._data["models"][i]

        # Deduplicate: if same rave_path already registered, update instead of adding
        if "rave_path" in entry:
            for i, m in enumerate(self._data["models"]):
                if m.get("rave_path") == entry["rave_path"]:
                    self._data["models"][i] = {**m, **entry}
                    if make_active:
                        self._data["active_model"] = m["id"]
                    self._save()
                    return self._data["models"][i]

        self._data["models"].append(entry)
        if make_active:
            self._data["active_model"] = model_id
        self._save()
        return entry

    def list_models(self) -> list[dict]:
        return list(self._data["models"])

    def get_model(self, model_id: str) -> dict | None:
        return next((m for m in self._data["models"] if m["id"] == model_id), None)

    def get_active(self) -> dict | None:
        aid = self._data.get("active_model")
        return self.get_model(aid) if aid else None

    def set_active(self, model_id: str) -> bool:
        if self.get_model(model_id):
            self._data["active_model"] = model_id
            self._save()
            return True
        return False

    def rename(self, model_id: str, new_name: str) -> bool:
        for m in self._data["models"]:
            if m["id"] == model_id:
                m["name"] = new_name
                self._save()
                return True
        return False

    def delete(self, model_id: str, delete_files: bool = False) -> bool:
        m = self.get_model(model_id)
        if not m:
            return False
        if delete_files:
            for key in ("vae_path", "lstm_path", "rave_path"):
                p = m.get(key)
                if p and Path(p).exists():
                    try:
                        Path(p).unlink()
                    except Exception:
                        pass
            for ckpt in m.get("checkpoints", []):
                p = ckpt.get("path")
                if p and Path(p).exists():
                    try:
                        Path(p).unlink()
                    except Exception:
                        pass
            # If there's a whole run dir, remove it
            run_dir = RUNS_DIR / model_id
            if run_dir.exists():
                shutil.rmtree(run_dir, ignore_errors=True)
        self._data["models"] = [x for x in self._data["models"] if x["id"] != model_id]
        if self._data.get("active_model") == model_id:
            self._data["active_model"] = (self._data["models"][-1]["id"]
                                           if self._data["models"] else None)
        self._save()
        return True

    @property
    def active_id(self) -> str | None:
        return self._data.get("active_model")


# ---------------------------------------------------------------------------
# In-memory model cache
# ---------------------------------------------------------------------------

class ModelCache:
    """Holds loaded models in memory to avoid re-loading on each generation."""

    def __init__(self):
        self._clear()

    def _clear(self):
        self.model_id:  str | None  = None
        self.backend:   str | None  = None
        self.vae                    = None
        self.lstm                   = None
        self.rave                   = None
        self.device:    str | None  = None
        self.vae_config: dict       = {}

    def is_loaded(self) -> bool:
        return self.vae is not None or self.rave is not None

    def info(self) -> dict:
        import torch
        vram = 0
        if torch.cuda.is_available() and self.is_loaded():
            vram = torch.cuda.memory_allocated() // (1024 * 1024)
        return {
            "loaded":   self.is_loaded(),
            "model_id": self.model_id,
            "backend":  self.backend,
            "device":   self.device,
            "vram_mb":  vram,
        }

    def load(self, model: dict, device_str: str | None = None) -> str:
        """Load a model dict from the registry into memory."""
        import torch
        self.unload()

        if device_str is None:
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_str)
        self.device   = device_str
        self.model_id = model["id"]
        self.backend  = model["backend"]

        if model["backend"] == "vae_lstm":
            from src.model import MusicVAE, LatentLSTM
            vae_path = model.get("vae_path")
            if not vae_path or not Path(vae_path).exists():
                raise FileNotFoundError(f"VAE checkpoint not found: {vae_path}")
            ckpt = torch.load(vae_path, map_location=device, weights_only=False)
            cfg  = ckpt.get("config", {})
            self.vae_config = cfg
            self.vae = MusicVAE(
                n_mels=cfg.get("n_mels", 128),
                n_frames=cfg.get("n_frames", 172),
                latent_dim=cfg.get("latent_dim", 128),
            ).to(device)
            self.vae.load_state_dict(ckpt["model_state_dict"])
            self.vae.eval()

            lstm_path = model.get("lstm_path")
            if lstm_path and Path(lstm_path).exists():
                ckpt2 = torch.load(lstm_path, map_location=device, weights_only=False)
                cfg2  = ckpt2.get("config", {})
                self.lstm = LatentLSTM(latent_dim=cfg2.get("latent_dim", 128)).to(device)
                self.lstm.load_state_dict(ckpt2["model_state_dict"])
                self.lstm.eval()

            return f"Loaded VAE+LSTM on {device_str}"

        elif model["backend"] == "rave":
            rave_path = model.get("rave_path")
            if not rave_path or not Path(rave_path).exists():
                raise FileNotFoundError(f"RAVE model not found: {rave_path}")
            self.rave = torch.jit.load(rave_path, map_location=device)
            self.rave.eval()
            return f"Loaded RAVE on {device_str}"

        else:
            raise ValueError(f"Unknown backend: {model['backend']}")

    def unload(self) -> None:
        import torch
        self._clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate(
        self,
        duration_s: float = 20.0,
        temperature: float = 0.5,
        griffin_lim_iters: int = 128,
        seed: int | None = None,
    ) -> str:
        """Generate audio using the currently cached model. Returns path to WAV."""
        if not self.is_loaded():
            raise RuntimeError("No model loaded — call load() first.")

        import numpy as np
        import torch
        from pathlib import Path
        import soundfile as sf
        import time as _time

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        OUTPUT_DIR = ROOT / "outputs"
        OUTPUT_DIR.mkdir(exist_ok=True)
        out_path = str(OUTPUT_DIR / f"gen_{int(_time.time())}.wav")

        if self.backend == "rave":
            return self._generate_rave(duration_s, temperature, out_path)
        else:
            return self._generate_vae(duration_s, temperature, griffin_lim_iters, out_path)

    def _generate_vae(self, duration_s, temperature, gl_iters, out_path) -> str:
        """Generation via VAE+LSTM pipeline."""
        import numpy as np, torch, soundfile as sf
        from src.audio_utils import denormalize, melspec_to_audio, SAMPLE_RATE, HOP_LENGTH, FRAMES_PER_CHUNK, CHUNK_DURATION
        from src.generate import _crossfade_mel, _slerp

        device     = torch.device(self.device or "cpu")
        latent_dim = self.vae_config.get("latent_dim", 128)
        n_chunks   = max(2, int(np.ceil(duration_s / CHUNK_DURATION)) + 1)

        with torch.no_grad():
            if self.lstm is not None:
                latents = self.lstm.generate(n_chunks, str(device), temperature)
            else:
                n_anchors = max(2, n_chunks // 8 + 2)
                anchors   = [torch.randn(latent_dim, device=device) * temperature
                             for _ in range(n_anchors)]
                codes     = []
                for t_g in np.linspace(0, n_anchors - 1, n_chunks):
                    si = min(int(t_g), n_anchors - 2)
                    codes.append(_slerp(anchors[si], anchors[si + 1], t_g - int(t_g)))
                latents = torch.stack(codes)

            all_mels = []
            for i in range(n_chunks):
                z   = latents[i].unsqueeze(0).to(device)
                mel = self.vae.decode(z).squeeze().cpu().numpy()
                all_mels.append(denormalize(mel))

        overlap = max(8, int(0.4 * SAMPLE_RATE / HOP_LENGTH))
        mel_full = _crossfade_mel(all_mels, overlap)
        audio    = melspec_to_audio(mel_full, n_iter=gl_iters)
        target   = int(duration_s * SAMPLE_RATE)
        audio    = audio[:target] if len(audio) >= target else np.pad(audio, (0, target - len(audio)))
        audio    = np.clip(audio / (np.abs(audio).max() + 1e-8), -1.0, 1.0)
        sf.write(out_path, audio, SAMPLE_RATE, subtype="PCM_16")
        return out_path

    def _generate_rave(self, duration_s, temperature, out_path) -> str:
        """Generation via RAVE model."""
        import torch, soundfile as sf, numpy as np
        model  = self.rave
        device = torch.device(self.device or "cpu")

        # sr is reliably exposed as an attribute
        try:
            sr = int(model.sr)
        except Exception:
            sr = 44100

        # Probe latent_dim and compression ratio by encoding a short dummy signal.
        # model.encode_params[0] returns 1 (batch size placeholder), NOT the hop.
        # The encode probe is the only reliable way to get both values.
        probe_len  = 8192
        try:
            with torch.no_grad():
                dummy  = torch.zeros(1, 1, probe_len, device=device)
                z_probe = model.encode(dummy)
            latent_dim = z_probe.shape[1]
            ratio      = max(1, probe_len // max(1, z_probe.shape[2]))
        except Exception:
            latent_dim = 1
            ratio      = 2048

        z_steps = max(1, int(duration_s * sr / ratio))
        z = torch.randn(1, latent_dim, z_steps, device=device) * temperature

        with torch.no_grad():
            y = model.decode(z)

        audio = y.squeeze().cpu().numpy()
        if audio.ndim > 1:
            audio = audio[0]
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / peak * 0.95

        sf.write(out_path, audio.astype(np.float32), sr)
        return out_path


# Module-level singletons
registry = ModelRegistry()
cache    = ModelCache()

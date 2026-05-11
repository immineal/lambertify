"""Flask web server for Lambertify — training dashboard + generation UI."""
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from flask import Flask, render_template, request, jsonify, send_file

from src.hardware import (
    query_gpus, query_cpu, recommend_params, estimate_vram_mb, vram_warning,
    recommend_preprocess, assess_data,
)
from src.logger import (
    new_training_run, log_generation, log_preprocess,
    load_runs, load_run, load_generations, load_events,
    TrainingRun,
)
from src.model_manager import registry as _registry, cache as _cache
import src.rave_backend as _rave
import src.config as _cfg
from src.data_integrity import check_stale, scan_dir

# Mark any runs that were "running" when the app last died as interrupted
def _mark_stale_runs() -> None:
    from src.logger import RUNS_DIR
    import json as _json
    for p in RUNS_DIR.glob("*_train.json"):
        try:
            with open(p) as f:
                d = _json.load(f)
            if d.get("status") == "running":
                d["status"]  = "interrupted"
                d["error"]   = "Server was restarted while training was active."
                with open(p, "w") as f:
                    _json.dump(d, f, indent=2)
        except Exception:
            pass

_mark_stale_runs()

OUTPUT_DIR     = Path(ROOT) / "outputs"
CHECKPOINT_DIR = Path(ROOT) / "checkpoints"
PROCESSED_DIR  = Path(ROOT) / "processed"
OUTPUT_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)

app = Flask(__name__, template_folder="templates", static_folder="static")

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

_state_lock = threading.Lock()
_jobs: dict[str, dict[str, Any]] = {}
_current_run: TrainingRun | None = None   # active training run (logged to disk)

_train: dict[str, Any] = {
    "status":       "idle",
    "stage":        "",
    "epoch":        0,
    "total_epochs": 0,
    "history": {"vae_train": [], "vae_val": [], "lstm_train": [], "lstm_val": []},
    "vram_mb":  0,
    "log":      [],
    "error":    None,
    "started":  None,
    "finished": None,
    "run_id":   None,
}


def _log(msg: str) -> None:
    ts   = time.strftime("%H:%M:%S")
    # Ensure every message is on its own line — prevents concatenation when
    # multiple callbacks fire in quick succession without intervening newlines
    line = f"[{ts}] {msg.strip()}"
    print(line)
    with _state_lock:
        _train["log"].append(line)
        if len(_train["log"]) > 500:          # cap in-memory log at 500 lines
            _train["log"] = _train["log"][-500:]
    if _current_run is not None:
        _current_run.log_line(line)


def _progress_cb(stage: str, epoch: int, total: int,
                 train_loss: float, val_loss: float, vram_mb: int) -> None:
    with _state_lock:
        _train["stage"]        = stage
        _train["epoch"]        = epoch
        _train["total_epochs"] = total
        _train["vram_mb"]      = vram_mb
        _train["history"][f"{stage}_train"].append(round(train_loss, 6))
        _train["history"][f"{stage}_val"].append(round(val_loss,   6))
    _log(f"[{stage.upper()}] {epoch}/{total}  "
         f"train={train_loss:.5f}  val={val_loss:.5f}  vram={vram_mb}MB")
    if _current_run is not None:
        _current_run.log_epoch(stage, epoch, total, train_loss, val_loss, vram_mb)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model_state() -> dict:
    active = _registry.get_active()
    preprocessed = (PROCESSED_DIR / "chunks.npy").exists()

    if active and active.get("backend") == "rave":
        rave_path = Path(active.get("rave_path", ""))
        ready = rave_path.exists()
        return {
            "vae_ready":      ready,   # "ready" means we can generate
            "lstm_ready":     False,
            "preprocessed":   preprocessed,
            "active_model":   active,
            "cache_loaded":   _cache.is_loaded(),
            "cache_model_id": _cache.model_id,
        }

    # VAE+LSTM path
    vae_path  = Path(active["vae_path"])       if (active and active.get("vae_path"))  else CHECKPOINT_DIR / "vae_best.pt"
    lstm_path = Path(active["lstm_path"])      if (active and active.get("lstm_path")) else CHECKPOINT_DIR / "lstm_best.pt"
    return {
        "vae_ready":      vae_path.exists(),
        "lstm_ready":     lstm_path.exists(),
        "preprocessed":   preprocessed,
        "active_model":   active,
        "cache_loaded":   _cache.is_loaded(),
        "cache_model_id": _cache.model_id,
    }


def _hw_snapshot() -> dict:
    gpus = query_gpus()
    cpu  = query_cpu()
    return {
        "gpus": [g.to_dict() for g in gpus],
        "cpu":  cpu.to_dict(),
    }


def _model_info() -> dict:
    """Best-effort metadata about the current checkpoint."""
    info: dict = {}
    try:
        import torch
        p = CHECKPOINT_DIR / "vae_best.pt"
        if p.exists():
            c = torch.load(p, map_location="cpu", weights_only=False)
            info["vae_epoch"]    = c.get("epoch")
            info["vae_val_loss"] = c.get("val_loss")
            info["vae_config"]   = c.get("config")
        p2 = CHECKPOINT_DIR / "lstm_best.pt"
        if p2.exists():
            c2 = torch.load(p2, map_location="cpu", weights_only=False)
            info["lstm_epoch"]    = c2.get("epoch")
            info["lstm_val_loss"] = c2.get("val_loss")
    except Exception:
        pass
    return info


# ---------------------------------------------------------------------------
# Routes — pages
# ---------------------------------------------------------------------------

def _data_dir() -> Path:
    """Return the configured data directory."""
    return Path(_cfg.get("data_dir", str(Path(ROOT) / "data")))


@app.route("/")
def index():
    gpus    = query_gpus()
    cpu     = query_cpu()
    data_dir = _data_dir()
    stats   = _dataset_stats()
    n_audio = len([f for f in data_dir.glob("*") if f.suffix.lower()
                   in (".mp3", ".wav", ".flac", ".ogg", ".aac", ".opus")])
    rec     = recommend_params(gpus[0] if gpus else None, dataset_stats=stats)
    prec    = recommend_preprocess(
        n_mp3_files=n_audio,
        current_n_chunks=stats.get("n_chunks") if stats else None,
    )
    cfg     = _cfg.load()
    return render_template(
        "index.html",
        model_state=_model_state(),
        gpus=[g.to_dict() for g in gpus],
        cpu=cpu.to_dict(),
        rec=rec,
        prec=prec,
        cfg=cfg,
        data_dir=str(data_dir),
    )


# ---------------------------------------------------------------------------
# Routes — config
# ---------------------------------------------------------------------------

@app.route("/api/config")
def api_config_get():
    return jsonify(_cfg.load())


@app.route("/api/config", methods=["POST"])
def api_config_set():
    d = request.get_json(force=True) or {}
    current = _cfg.load()
    # Allow updating top-level keys and nested sections
    for k, v in d.items():
        if isinstance(v, dict) and isinstance(current.get(k), dict):
            current[k] = {**current.get(k, {}), **v}
        else:
            current[k] = v
    _cfg.save(current)
    return jsonify({"ok": True, "config": current})


# ---------------------------------------------------------------------------
# Routes — data status
# ---------------------------------------------------------------------------

@app.route("/api/data_status")
def api_data_status():
    """Return data directory contents + freshness of VAE and RAVE preprocessing."""
    data_dir = _data_dir()
    scan     = scan_dir(data_dir)

    vae_check  = check_stale(data_dir, PROCESSED_DIR)
    rave_check = check_stale(data_dir, _rave.RAVE_DATA_DIR)

    return jsonify({
        "data_dir":        str(data_dir),
        "scan":            scan,
        "vae_stale":       vae_check["stale"],
        "vae_stale_reason": vae_check["reason"],
        "vae_snapshot":    vae_check["snapshot"],
        "rave_stale":      rave_check["stale"],
        "rave_stale_reason": rave_check["reason"],
        "rave_snapshot":   rave_check["snapshot"],
        **_model_state(),
    })


# ---------------------------------------------------------------------------
# Routes — hardware
# ---------------------------------------------------------------------------

@app.route("/api/hw")
def api_hw():
    gpus = query_gpus()
    cpu  = query_cpu()
    return jsonify({"gpus": [g.to_dict() for g in gpus], "cpu": cpu.to_dict()})


def _dataset_stats() -> dict | None:
    """Load processed/stats.json if available."""
    stats_path = PROCESSED_DIR / "stats.json"
    if not stats_path.exists():
        return None
    try:
        with open(stats_path) as f:
            return json.load(f)
    except Exception:
        return None


@app.route("/api/params")
def api_params():
    gpus   = query_gpus()
    rec    = recommend_params(gpus[0] if gpus else None, dataset_stats=_dataset_stats())
    return jsonify(rec)


@app.route("/api/preprocess_recommend")
def api_preprocess_recommend():
    """Recommended preprocessing settings for the current dataset."""
    stats = _dataset_stats()
    gpus  = query_gpus()
    n_mp3 = len(list((Path(ROOT) / "data").glob("*.mp3")))
    rec   = recommend_preprocess(
        n_mp3_files=n_mp3,
        current_n_chunks=stats.get("n_chunks") if stats else None,
    )
    return jsonify(rec)


@app.route("/api/assess_data")
def api_assess_data():
    """Data adequacy assessment and prioritised action list."""
    stats = _dataset_stats()
    if not stats:
        return jsonify({"tier": "unknown", "summary": "Preprocessing not run yet.",
                        "actions": [], "n_chunks": 0, "n_songs": 0})
    hop = stats.get("hop_fraction", 0.5)
    return jsonify(assess_data(
        n_chunks=stats.get("n_chunks", 0),
        n_songs=stats.get("n_songs", 0),
        hop_fraction=hop,
    ))


@app.route("/api/estimate_vram", methods=["POST"])
def api_estimate_vram():
    d          = request.get_json(force=True) or {}
    vae_batch  = int(d.get("vae_batch_size",  32))
    lstm_batch = int(d.get("lstm_batch_size", 128))
    latent_dim = int(d.get("latent_dim",      128))
    seq_len    = int(d.get("seq_len",          16))
    est        = estimate_vram_mb(vae_batch, latent_dim, lstm_batch, seq_len)
    gpus       = query_gpus()
    free       = gpus[0].vram_free_mb if gpus else 0
    warns      = []
    for stage, mb in (("VAE", est["vae_mb"]), ("LSTM", est["lstm_mb"])):
        w = vram_warning(mb, free, stage)
        if w:
            warns.append(w)
    return jsonify({**est, "vram_free_mb": free, "warnings": warns})


# ---------------------------------------------------------------------------
# Routes — preprocessing
# ---------------------------------------------------------------------------

@app.route("/api/preprocess", methods=["POST"])
def api_preprocess():
    with _state_lock:
        if _train["status"] in ("preprocessing", "training"):
            return jsonify({"error": "Already running"}), 409

    d            = request.get_json(force=True) or {}
    hop_fraction = max(0.1, min(0.9, float(d.get("hop_fraction", 0.5))))

    data_dir = str(_data_dir())

    def _run():
        t0 = time.time()
        _log(f"VAE preprocessing started — data_dir={data_dir}, hop_fraction={hop_fraction}")
        with _state_lock:
            _train["status"] = "preprocessing"
            _train["error"]  = None
        try:
            from src.preprocess import preprocess_all
            import src.preprocess as _pp
            _pp.DATA_DIR = Path(data_dir)   # use configured dir
            result = preprocess_all(hop_fraction=hop_fraction, log_cb=_log)
            elapsed = round(time.time() - t0, 1)
            stats_path = PROCESSED_DIR / "stats.json"
            stats = {}
            if stats_path.exists():
                with open(stats_path) as f:
                    stats = json.load(f)
            _log(f"Preprocessing complete in {elapsed}s — {stats.get('n_chunks', '?')} chunks")
            log_preprocess(
                params={"hop_fraction": hop_fraction},
                result=stats,
                status="done",
            )
            with _state_lock:
                _train["status"] = "idle"
        except Exception as exc:
            import traceback
            err = traceback.format_exc()
            _log(f"Preprocessing failed: {exc}")
            log_preprocess(
                params={"hop_fraction": hop_fraction},
                status="error",
                error=err,
            )
            with _state_lock:
                _train["status"] = "error"
                _train["error"]  = str(exc)

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/preprocess_stats")
def api_preprocess_stats():
    stats_path = PROCESSED_DIR / "stats.json"
    if not stats_path.exists():
        return jsonify({"ready": False})
    with open(stats_path) as f:
        stats = json.load(f)
    stats["ready"] = True
    return jsonify(stats)


# ---------------------------------------------------------------------------
# Routes — training
# ---------------------------------------------------------------------------

@app.route("/api/train", methods=["POST"])
def api_train():
    global _current_run
    with _state_lock:
        if _train["status"] in ("preprocessing", "training"):
            return jsonify({"error": "Already running"}), 409

    d = request.get_json(force=True) or {}
    import time as _t
    params = {
        "stage":            d.get("stage",           "both"),
        "vae_epochs":       int(d.get("vae_epochs",  50)),
        "lstm_epochs":      int(d.get("lstm_epochs", 30)),
        "vae_batch_size":   int(d.get("vae_batch_size",  32)),
        "lstm_batch_size":  int(d.get("lstm_batch_size", 128)),
        "latent_dim":       int(d.get("latent_dim",  128)),
        "lr_vae":           float(d.get("lr_vae",    2e-4)),
        "lr_lstm":          float(d.get("lr_lstm",   1e-3)),
        "kl_max":           float(d.get("kl_max",    5e-4)),
        "kl_ramp_pct":      float(d.get("kl_ramp_pct", 0.4)),
        "seq_len":          int(d.get("seq_len",     16)),
        "num_workers":      int(d.get("num_workers", 2)),
        "device":           str(d.get("device", "cuda" if __import__("torch").cuda.is_available() else "cpu")),
        "checkpoint_every": int(d.get("checkpoint_every", 50)),
        "run_id":           d.get("run_id") or _t.strftime("%Y%m%d_%H%M%S"),
    }

    hw = _hw_snapshot()
    _current_run = new_training_run(params, hardware=hw)

    with _state_lock:
        _train.update(
            status="training", stage="", epoch=0, total_epochs=0,
            vram_mb=0, error=None,
            started=time.strftime("%Y-%m-%d %H:%M:%S"), finished=None,
            run_id=_current_run.run_id,
            history={"vae_train": [], "vae_val": [], "lstm_train": [], "lstm_val": []},
        )
        _train["log"].clear()

    def _run():
        global _current_run
        try:
            from src.train import run_training
            run_training(
                stage=params["stage"],
                vae_epochs=params["vae_epochs"],
                lstm_epochs=params["lstm_epochs"],
                vae_batch_size=params["vae_batch_size"],
                lstm_batch_size=params["lstm_batch_size"],
                latent_dim=params["latent_dim"],
                lr_vae=params["lr_vae"],
                lr_lstm=params["lr_lstm"],
                kl_max=params["kl_max"],
                kl_ramp_pct=params["kl_ramp_pct"],
                seq_len=params["seq_len"],
                num_workers=params["num_workers"],
                device_str=params["device"],
                progress_cb=_progress_cb,
                checkpoint_every=params["checkpoint_every"],
                run_id=params["run_id"],
            )
            _log("Training finished successfully")
            _current_run.finish("done")
            with _state_lock:
                _train["status"]   = "done"
                _train["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        except Exception as exc:
            import traceback
            err = traceback.format_exc()
            _log(f"Training error: {exc}")
            _current_run.finish("error", error=err)
            with _state_lock:
                _train["status"]   = "error"
                _train["error"]    = err
                _train["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        finally:
            _current_run = None

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"ok": True, "run_id": params.get("stage")})


@app.route("/api/train_status")
def api_train_status():
    with _state_lock:
        snap = {
            "status":       _train["status"],
            "stage":        _train["stage"],
            "epoch":        _train["epoch"],
            "total_epochs": _train["total_epochs"],
            "vram_mb":      _train["vram_mb"],
            "history":      _train["history"],
            "error":        _train["error"],
            "started":      _train["started"],
            "finished":     _train["finished"],
            "run_id":       _train.get("run_id"),
            **_model_state(),
        }
    return jsonify(snap)


@app.route("/api/train_log")
def api_train_log():
    n = int(request.args.get("n", 200))
    with _state_lock:
        lines = list(_train["log"][-n:])
    return jsonify({"lines": lines})


@app.route("/api/train_stop", methods=["POST"])
def api_train_stop():
    with _state_lock:
        if _train["status"] == "training":
            _train["status"] = "idle"
            _train["log"].append("[WARN] Stop requested — current epoch will finish")
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Routes — generation
# ---------------------------------------------------------------------------

@app.route("/api/generate", methods=["POST"])
def api_generate():
    if not _model_state()["vae_ready"]:
        return jsonify({"error": "Model not trained yet. Train first."}), 503

    d           = request.get_json(force=True) or {}
    duration    = max(5.0,  min(300.0, float(d.get("duration",        20))))
    temperature = max(0.1,  min(3.0,   float(d.get("temperature",    1.0))))
    gl_iters    = max(16,   min(512,   int(d.get("griffin_lim_iters", 64))))
    seed_raw    = d.get("seed")
    seed        = int(seed_raw) if seed_raw not in (None, "", "null") else None

    job_id = str(uuid.uuid4())
    with _state_lock:
        _jobs[job_id] = {"status": "pending", "file": None, "error": None}

    gen_params = {
        "duration":          duration,
        "temperature":       temperature,
        "griffin_lim_iters": gl_iters,
        "seed":              seed,
    }

    def _run():
        t0 = time.time()
        with _state_lock:
            _jobs[job_id]["status"] = "running"
        out_path = str(OUTPUT_DIR / f"{job_id}.wav")
        try:
            if _cache.is_loaded():
                import numpy as np, torch
                if seed is not None:
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                _cache._generate_vae(duration, temperature, gl_iters, out_path) \
                    if _cache.backend == "vae_lstm" \
                    else _cache._generate_rave(duration, temperature, out_path)
            else:
                from src.generate import generate_audio
                generate_audio(
                    duration_s=duration,
                    output_path=out_path,
                    temperature=temperature,
                    griffin_lim_iters=gl_iters,
                    seed=seed,
                )
            elapsed = time.time() - t0
            log_generation(
                params=gen_params,
                output_path=out_path,
                gen_time_s=elapsed,
                model_info=_model_info(),
                status="done",
            )
            with _state_lock:
                _jobs[job_id]["status"] = "done"
                _jobs[job_id]["file"]   = out_path
                _jobs[job_id]["gen_time_s"] = round(elapsed, 1)
        except Exception as exc:
            import traceback
            err = traceback.format_exc()
            log_generation(
                params=gen_params,
                output_path=out_path,
                gen_time_s=time.time() - t0,
                model_info=_model_info(),
                status="error",
                error=err,
            )
            with _state_lock:
                _jobs[job_id]["status"] = "error"
                _jobs[job_id]["error"]  = err

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/api/job/<job_id>")
def api_job(job_id: str):
    with _state_lock:
        job = _jobs.get(job_id)
    if not job:
        return jsonify({"status": "not_found"}), 404
    return jsonify({"status": job["status"], "error": job.get("error"),
                    "gen_time_s": job.get("gen_time_s")})


@app.route("/api/audio/<job_id>")
def api_audio(job_id: str):
    with _state_lock:
        job = _jobs.get(job_id)
    if not job or job["status"] != "done":
        return jsonify({"error": "not ready"}), 404
    return send_file(job["file"], mimetype="audio/wav")


# Backwards-compat aliases
@app.route("/generate", methods=["POST"])
def compat_generate(): return api_generate()
@app.route("/job/<job_id>")
def compat_job(job_id): return api_job(job_id)
@app.route("/audio/<job_id>")
def compat_audio(job_id): return api_audio(job_id)


@app.route("/api/outputs")
def api_outputs():
    files = sorted(OUTPUT_DIR.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    return jsonify([
        {"name": p.name, "url": f"/api/outputs/{p.name}",
         "size_mb": round(p.stat().st_size / 1e6, 2), "ts": int(p.stat().st_mtime)}
        for p in files[:50]
    ])


@app.route("/api/outputs/<filename>")
def api_output_file(filename: str):
    path = OUTPUT_DIR / filename
    if not path.exists() or path.suffix != ".wav":
        return "Not found", 404
    return send_file(str(path), mimetype="audio/wav")


# ---------------------------------------------------------------------------
# Routes — model management
# ---------------------------------------------------------------------------

@app.route("/api/models")
def api_models():
    models  = _registry.list_models()
    active  = _registry.active_id
    cache_i = _cache.info()
    return jsonify({"models": models, "active_id": active, "cache": cache_i})


@app.route("/api/models/activate", methods=["POST"])
def api_models_activate():
    model_id = (request.get_json(force=True) or {}).get("model_id")
    if not model_id:
        return jsonify({"error": "model_id required"}), 400
    if not _registry.set_active(model_id):
        return jsonify({"error": "model not found"}), 404
    return jsonify({"ok": True, "active_id": model_id})


@app.route("/api/models/load", methods=["POST"])
def api_models_load():
    d        = request.get_json(force=True) or {}
    model_id = d.get("model_id") or (_registry.active_id)
    device   = d.get("device")
    if not model_id:
        return jsonify({"error": "no model_id and no active model"}), 400
    m = _registry.get_model(model_id)
    if not m:
        return jsonify({"error": "model not found"}), 404
    try:
        msg = _cache.load(m, device_str=device)
        _log(f"Model loaded: {m['name']} — {msg}")
        return jsonify({"ok": True, "msg": msg, "cache": _cache.info()})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/models/unload", methods=["POST"])
def api_models_unload():
    _cache.unload()
    _log("Model unloaded from memory.")
    return jsonify({"ok": True, "cache": _cache.info()})


@app.route("/api/models/rename", methods=["POST"])
def api_models_rename():
    d = request.get_json(force=True) or {}
    if _registry.rename(d.get("model_id", ""), d.get("name", "")):
        return jsonify({"ok": True})
    return jsonify({"error": "not found"}), 404


@app.route("/api/models/delete", methods=["POST"])
def api_models_delete():
    d            = request.get_json(force=True) or {}
    model_id     = d.get("model_id", "")
    delete_files = bool(d.get("delete_files", False))
    if _registry.delete(model_id, delete_files=delete_files):
        if _cache.model_id == model_id:
            _cache.unload()
        return jsonify({"ok": True})
    return jsonify({"error": "not found"}), 404


@app.route("/api/models/cache_info")
def api_models_cache_info():
    return jsonify(_cache.info())


# ---------------------------------------------------------------------------
# Routes — generation (update to use cache when loaded)
# ---------------------------------------------------------------------------

def _generate_with_best_logic(duration, temperature, gl_iters, seed, out_path):
    """Generate using cached model if available, otherwise load from active registry."""
    if _cache.is_loaded():
        return _cache._generate_vae(duration, temperature, gl_iters, out_path) \
               if _cache.backend == "vae_lstm" \
               else _cache._generate_rave(duration, temperature, out_path)

    # Fall back to the legacy file-based approach
    from src.generate import generate_audio
    return generate_audio(
        duration_s=duration, output_path=out_path,
        temperature=temperature, griffin_lim_iters=gl_iters, seed=seed,
    )


# ---------------------------------------------------------------------------
# Routes — RAVE backend
# ---------------------------------------------------------------------------

_rave_state: dict = {
    "status":     "idle",   # idle | installing | preprocessing | training | done | error
    "step":       0,
    "total_steps": 0,
    "log":        [],
    "error":      None,
    "model_path": None,
    "preprocess_ready": False,
}
_rave_stop_event = threading.Event()


def _rave_step_cb(step: int) -> None:
    _rave_state["step"] = step


def _rave_log(msg: str) -> None:
    ts   = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg.strip()}"
    print(line)
    _rave_state["log"].append(line)
    if len(_rave_state["log"]) > 500:
        _rave_state["log"] = _rave_state["log"][-500:]


@app.route("/api/rave/status")
def api_rave_status():
    installed = _rave.is_installed()
    # Only check install_error when explicitly requested (it spawns a subprocess)
    check_err = request.args.get("check_error") == "1"
    install_err = _rave.install_error() if (not installed and check_err) else None
    return jsonify({
        "installed":        installed,
        "install_error":    install_err,
        "status":           _rave_state["status"],
        "step":             _rave_state["step"],
        "total_steps":      _rave_state["total_steps"],
        "error":            _rave_state["error"],
        "model_path":       _rave_state["model_path"],
        "preprocess_ready": _rave.preprocess_ready(str(_data_dir())),
        "preprocess_info":  _rave.preprocess_info(),
        "vram_estimate":    _rave.estimate_vram_mb(
            batch_size=int(request.args.get("batch", 8))
        ),
        "log_tail":         _rave_state["log"][-100:],
    })


@app.route("/api/rave/params")
def api_rave_params():
    gpus   = query_gpus()
    stats  = _dataset_stats()
    hours  = (stats.get("n_chunks", 0) * stats.get("hop_fraction", 0.5) * 4 / 3600) if stats else 6.35
    return jsonify(_rave.recommend_params(
        gpu=gpus[0] if gpus else None,
        n_songs=stats.get("n_songs", 137) if stats else 137,
        hours_of_audio=hours,
    ))


@app.route("/api/rave/install", methods=["POST"])
def api_rave_install():
    if _rave.is_installed():
        return jsonify({"ok": True, "msg": "Already installed."})
    if _rave_state["status"] == "installing":
        return jsonify({"error": "Already installing"}), 409

    _rave_state["status"] = "installing"

    def _run():
        ok = _rave.install(log_cb=_rave_log)
        _rave_state["status"] = "idle" if ok else "error"
        if ok:
            _rave.invalidate_install_cache()
        else:
            _rave_state["error"] = "pip install acids-rave failed"

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/rave/preprocess", methods=["POST"])
def api_rave_preprocess():
    if not _rave.is_installed():
        return jsonify({"error": "RAVE not installed"}), 503
    if _rave_state["status"] in ("preprocessing", "training"):
        return jsonify({"error": "Already running"}), 409

    d        = request.get_json(force=True) or {}
    sr       = int(d.get("sample_rate", 44100))
    data_dir = str(_data_dir())
    _rave_state["status"] = "preprocessing"
    _rave_state["log"].clear()
    _rave_log(f"RAVE preprocessing started — data_dir={data_dir}, sample_rate={sr}")

    def _run():
        try:
            _rave.preprocess(data_dir=data_dir, sample_rate=sr, log_cb=_rave_log)
            _rave_log("RAVE preprocessing complete")
            _rave_state["status"] = "idle"
        except Exception as exc:
            import traceback
            _rave_log(f"Preprocessing failed: {exc}")
            _rave_state["status"] = "error"
            _rave_state["error"]  = traceback.format_exc()

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/rave/train", methods=["POST"])
def api_rave_train():
    if not _rave.is_installed():
        return jsonify({"error": "RAVE not installed — click Install first"}), 503
    if _rave_state["status"] == "training":
        return jsonify({"error": "Already training"}), 409

    d = request.get_json(force=True) or {}
    import src.rave_backend as _rb
    params = {
        "name":        d.get("name",    "lambert"),
        "config":      d.get("config",  "v2"),
        "n_signal":    _rb.RAVE_NUM_SIGNAL,          # always 131072 — must match preprocessing
        "batch_size":  int(d.get("batch_size", 8)),
        "n_steps":     int(d.get("n_steps",    500_000)),
        "workers":     int(d.get("workers",    4)),
        "sample_rate": int(d.get("sample_rate", 44100)),
    }

    _rave_state.update(
        status="training", step=0, total_steps=params["n_steps"],
        error=None, model_path=None,
    )
    _rave_state["log"].clear()
    _rave_stop_event.clear()
    _rave_log(f"Starting RAVE training — {params}")

    def _run():
        try:
            model_path = _rave.train(
                **params,
                log_cb=_rave_log,
                step_cb=_rave_step_cb,
                stop_event=_rave_stop_event,
            )
            if model_path:
                _rave_state["model_path"] = model_path
                _rave_log(f"Training complete → {model_path}")
                _rave_state["status"] = "done"
                # Register in model registry
                model_id = f"rave_{params['name']}_{time.strftime('%Y%m%d_%H%M%S')}"
                _registry.register_rave(
                    model_id=model_id,
                    model_path=model_path,
                    config=params,
                    name=f"RAVE {params['config']} — {params['name']}",
                    make_active=True,
                )
            else:
                _rave_state["status"] = "error"
                _rave_state["error"]  = "Export failed — no .ts file found"
        except Exception as exc:
            import traceback
            _rave_state["status"] = "error"
            _rave_state["error"]  = traceback.format_exc()
            _rave_log(f"RAVE training error: {exc}")

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/rave/stop", methods=["POST"])
def api_rave_stop():
    _rave_stop_event.set()
    _rave.stop_training()
    _rave_log("Stop requested — exporting current checkpoint...")
    return jsonify({"ok": True})


@app.route("/api/rave/log")
def api_rave_log():
    n = int(request.args.get("n", 100))
    return jsonify({"lines": _rave_state["log"][-n:]})


# ---------------------------------------------------------------------------
# Routes — history
# ---------------------------------------------------------------------------

@app.route("/api/history/runs")
def api_history_runs():
    """List of all training runs (summary), newest first."""
    return jsonify(load_runs(limit=100))


@app.route("/api/history/run/<run_id>")
def api_history_run(run_id: str):
    """Full data for one training run."""
    data = load_run(run_id)
    if data is None:
        return jsonify({"error": "not found"}), 404
    return jsonify(data)


@app.route("/api/history/generations")
def api_history_generations():
    """List of all generation records, newest first."""
    return jsonify(load_generations(limit=200))


@app.route("/api/history/events")
def api_history_events():
    """Most recent events from events.jsonl."""
    limit = int(request.args.get("n", 100))
    return jsonify(load_events(limit=limit))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host",  default="0.0.0.0")
    parser.add_argument("--port",  type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug,
            use_reloader=False, threaded=True)

"""Structured persistent logging for training runs, generation, and preprocessing.

Everything is stored in logs/ as JSON / JSONL files that survive server restarts:

  logs/
    events.jsonl           – append-only stream of all significant events
    generations.jsonl      – one entry per generation (params + outcome)
    runs/
      20260511_142301_train.json   – full detail for each training run
      ...

TrainingRun is the main object: instantiate it at training start, call
log_epoch() from the progress callback, and finish() when done.
The JSON file is written incrementally (every 10 epochs) so a crash still
leaves a readable partial record.
"""

import json
import os
import time
from pathlib import Path
from typing import Any

ROOT     = Path(__file__).parent.parent
LOGS_DIR = ROOT / "logs"
RUNS_DIR = LOGS_DIR / "runs"
LOGS_DIR.mkdir(exist_ok=True)
RUNS_DIR.mkdir(exist_ok=True)

EVENTS_FILE = LOGS_DIR / "events.jsonl"
GENS_FILE   = LOGS_DIR / "generations.jsonl"


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _append_jsonl(path: Path, obj: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _ts() -> float:
    return time.time()


def _ts_str(ts: float | None = None) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts or time.time()))


# ---------------------------------------------------------------------------
# Training runs
# ---------------------------------------------------------------------------

class TrainingRun:
    """Tracks one full training run (VAE + LSTM or either stage).

    Call log_epoch() from the training progress callback.
    Call finish() when training ends (success or error).
    The .json file is kept up to date throughout so partial runs are readable.
    """

    def __init__(self, params: dict, hardware: dict | None = None):
        self.run_id     = time.strftime("%Y%m%d_%H%M%S")
        self.ts_start   = _ts()
        self.params     = params
        self.hardware   = hardware or {}
        self.epochs: list[dict]    = []
        self.vram_log: list[dict]  = []
        self.log_lines: list[str]  = []
        self.status     = "running"
        self.error: str | None     = None
        self.ts_end: float | None  = None
        self.path       = RUNS_DIR / f"{self.run_id}_train.json"

        self._save()
        _append_jsonl(EVENTS_FILE, {
            "ts": self.ts_start, "ts_str": _ts_str(self.ts_start),
            "kind": "train_start", "run_id": self.run_id, "params": params,
        })

    def log_epoch(
        self,
        stage: str,
        epoch: int,
        total: int,
        train_loss: float,
        val_loss: float,
        vram_mb: int,
    ) -> None:
        entry = {
            "ts":         _ts(),
            "stage":      stage,
            "epoch":      epoch,
            "total":      total,
            "train_loss": round(train_loss, 7),
            "val_loss":   round(val_loss,   7),
            "vram_mb":    vram_mb,
        }
        self.epochs.append(entry)
        self.vram_log.append({"ts": entry["ts"], "vram_mb": vram_mb})
        if len(self.epochs) % 10 == 0:
            self._save()

    def log_line(self, line: str) -> None:
        self.log_lines.append(line)
        if len(self.log_lines) % 50 == 0:
            self._save()

    def finish(self, status: str = "done", error: str | None = None) -> None:
        self.status = status
        self.error  = error
        self.ts_end = _ts()
        self._save()
        _append_jsonl(EVENTS_FILE, {
            "ts": self.ts_end, "ts_str": _ts_str(self.ts_end),
            "kind":       "train_end",
            "run_id":     self.run_id,
            "status":     status,
            "duration_s": round(self.ts_end - self.ts_start),
            "n_epochs":   len(self.epochs),
            "error":      error,
        })

    def _save(self) -> None:
        best_vae  = min((e["val_loss"] for e in self.epochs if e["stage"] == "vae"),  default=None)
        best_lstm = min((e["val_loss"] for e in self.epochs if e["stage"] == "lstm"), default=None)
        data = {
            "run_id":        self.run_id,
            "ts_start":      self.ts_start,
            "ts_start_str":  _ts_str(self.ts_start),
            "params":        self.params,
            "hardware":      self.hardware,
            "status":        self.status,
            "epochs":        self.epochs,
            "vram_log":      self.vram_log[-200:],   # last 200 VRAM samples
            "log_lines":     self.log_lines[-500:],  # last 500 log lines
            "best_vae_val":  round(best_vae,  7) if best_vae  is not None else None,
            "best_lstm_val": round(best_lstm, 7) if best_lstm is not None else None,
            "error":         self.error,
        }
        if self.ts_end:
            data["ts_end"]      = self.ts_end
            data["ts_end_str"]  = _ts_str(self.ts_end)
            data["duration_s"]  = round(self.ts_end - self.ts_start)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


def new_training_run(params: dict, hardware: dict | None = None) -> TrainingRun:
    return TrainingRun(params, hardware)


# ---------------------------------------------------------------------------
# Generation logging
# ---------------------------------------------------------------------------

def log_generation(
    params: dict,
    output_path: str,
    gen_time_s: float,
    model_info: dict | None = None,
    status: str = "done",
    error: str | None = None,
) -> None:
    """Append one generation record to logs/generations.jsonl."""
    ts = _ts()
    entry = {
        "ts":         ts,
        "ts_str":     _ts_str(ts),
        "params":     params,
        "output":     output_path,
        "filename":   Path(output_path).name if output_path else None,
        "gen_time_s": round(gen_time_s, 1),
        "model":      model_info or {},
        "status":     status,
        "error":      error,
    }
    _append_jsonl(GENS_FILE, entry)
    _append_jsonl(EVENTS_FILE, {
        "ts": ts, "ts_str": _ts_str(ts),
        "kind": "generation", "status": status,
        "output": Path(output_path).name if output_path else None,
        **{k: v for k, v in params.items()},
    })


# ---------------------------------------------------------------------------
# Preprocessing logging
# ---------------------------------------------------------------------------

def log_preprocess(
    params: dict,
    result: dict | None = None,
    status: str = "done",
    error: str | None = None,
) -> None:
    ts = _ts()
    _append_jsonl(EVENTS_FILE, {
        "ts":     ts,
        "ts_str": _ts_str(ts),
        "kind":   "preprocess",
        "params": params,
        "result": result,
        "status": status,
        "error":  error,
    })


# ---------------------------------------------------------------------------
# Reading logs back
# ---------------------------------------------------------------------------

def load_runs(limit: int = 50) -> list[dict]:
    """Return training run summaries sorted newest-first."""
    runs = []
    for f in sorted(RUNS_DIR.glob("*_train.json"), reverse=True)[:limit]:
        try:
            with open(f, encoding="utf-8") as fh:
                d = json.load(fh)
            runs.append({
                "run_id":        d.get("run_id", f.stem),
                "ts_start_str":  d.get("ts_start_str", "?"),
                "ts_end_str":    d.get("ts_end_str"),
                "status":        d.get("status", "?"),
                "duration_s":    d.get("duration_s"),
                "params":        d.get("params", {}),
                "hardware":      d.get("hardware", {}),
                "best_vae_val":  d.get("best_vae_val"),
                "best_lstm_val": d.get("best_lstm_val"),
                "n_epochs":      len(d.get("epochs", [])),
                "error_summary": (d.get("error") or "")[:200] or None,
            })
        except Exception:
            pass
    return runs


def load_run(run_id: str) -> dict | None:
    """Return the full data for one training run."""
    path = RUNS_DIR / f"{run_id}_train.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_generations(limit: int = 100) -> list[dict]:
    """Return generation records newest-first."""
    if not GENS_FILE.exists():
        return []
    entries = []
    with open(GENS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return list(reversed(entries[-limit:]))


def load_events(limit: int = 200) -> list[dict]:
    """Return the most recent events from events.jsonl."""
    if not EVENTS_FILE.exists():
        return []
    entries = []
    with open(EVENTS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return list(reversed(entries[-limit:]))

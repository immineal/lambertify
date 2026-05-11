# Lambertify

A local music generation system trained on Lambert's ambient piano recordings.  
Two independent backends, one web UI.

---

## Backends

### VAE + LSTM
A convolutional Variational Autoencoder learns to encode 4-second mel-spectrogram chunks into a latent space.  A two-layer LSTM is then trained as an autoregressive prior over sequences of latent codes, giving temporal coherence across chunks.  At generation time, the LSTM samples a latent sequence, the VAE decoder converts each code back to a spectrogram, adjacent chunks are crossfaded in spectrogram space, and a single Griffin-Lim pass voices the result.

### RAVE
[RAVE](https://github.com/acids-icam/RAVE) (Realtime Audio Variational autoEncoder) by IRCAM ACIDS.  Operates directly on waveforms using an adversarially-trained encoder–decoder.  Produces dramatically better audio than Griffin-Lim.  Requires ~4–16 hours of training on an RTX 3070; 500 k steps gives recognisable Lambert style.

---

## Requirements

- Python 3.10+  
- CUDA-capable GPU (tested on RTX 3070, 8 GB VRAM)  
- openSUSE / any Linux with `nvidia-smi` on PATH

```
pip install -r requirements.txt          # VAE+LSTM pipeline
pip install acids-rave                   # RAVE backend (optional)
```

> **Note:** `acids-rave` requires three compatibility patches for scipy 1.14+ and numpy 2.x.  
> The app applies these automatically the first time RAVE is installed.

---

## Quick start

```bash
python main.py          # starts the web UI at http://localhost:5000
```

### Workflow

1. **Data tab** — point the folder picker at your audio directory (MP3 / WAV / FLAC).  
   Run *VAE preprocessing* (mel chunks) and/or *RAVE preprocessing* (LMDB) as needed.  
   Both show a freshness indicator: green = current, orange = re-run required.

2. **Train tab** — choose *VAE + LSTM* or *RAVE*, hit *⚡ Smart defaults* to fill in  
   GPU-aware hyperparameters, then start training.  
   Live loss chart, VRAM gauge, and scrollable log update in real time.

3. **Generate tab** — select duration and temperature, click *Generate*.  
   The active model (shown in the top banner) is used automatically.

4. **Models tab** — browse all trained checkpoints, load/unload into GPU memory,  
   activate a different model for generation, rename or delete old runs.

5. **History tab** — every training run and generation is logged to `logs/` as structured  
   JSON/JSONL with full hyperparameters, per-epoch losses, VRAM traces, and error  
   tracebacks if anything failed.

---

## Project structure

```
src/
  model.py          MusicVAE + LatentLSTM
  train.py          Two-stage training pipeline
  generate.py       VAE+LSTM generation (mel crossfade → Griffin-Lim)
  dataset.py        Mel chunk + latent sequence datasets, pitch augmentation
  audio_utils.py    Spectrogram ↔ audio helpers
  hardware.py       GPU/CPU detection, smart hyperparameter recommendations
  rave_backend.py   RAVE subprocess wrapper (preprocess / train / export / generate)
  model_manager.py  Checkpoint registry + in-memory model cache
  logger.py         Structured persistent logging
  config.py         User config persistence (data folder, last-used params)
  data_integrity.py Preprocessing freshness detection via file snapshots

web/
  app.py            Flask server — all API endpoints
  templates/        Jinja2 HTML
  static/           CSS + JS (Chart.js for loss curves)

main.py             Entry point
requirements.txt
```

---

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).

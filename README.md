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

## Training guide

### Choosing a backend

| | VAE + LSTM | RAVE |
|---|---|---|
| Audio quality | Medium — Griffin-Lim phase artefacts are always audible | High — waveform GAN, no Griffin-Lim |
| Training time | Fast (minutes–hours) | Slow (hours–days) |
| Minimum data | ~5 h usable | ~2 h usable |
| Sweet-spot data | 20–50 h | 6–20 h |
| Control over output | Duration, temperature, Griffin-Lim quality | Duration, temperature |
| Good for | Experimenting, iterating quickly | Final quality output |

Use **VAE + LSTM** while iterating on your dataset.  Switch to **RAVE** when you want the best audio quality and are willing to train overnight.

---

### Data

Both backends improve monotonically with more data.  The minimum for anything recognisably musical is roughly 2–3 hours; 6–12 hours is the sweet spot on a single-GPU setup.

**Getting more from the same files**

- **Reduce hop_fraction** (Data tab → VAE preprocessing).  The default of 0.5 means adjacent 4-second chunks overlap by 2 seconds.  Setting it to 0.2 gives 2.5× more chunks from the same audio — the most impactful free improvement available.  Re-run preprocessing after changing it.
- **Pitch augmentation** is applied automatically during VAE training (±3 mel bins, 60 % of batches).  It costs nothing and effectively multiplies dataset variety by ~5×.
- **Transposing files** with `sox` gives clean pitch-shifted copies that the model treats as independent songs:
  ```bash
  for f in data/*.mp3; do
      sox "$f" "${f%.mp3}_+2.mp3"  pitch 200
      sox "$f" "${f%.mp3}_-2.mp3"  pitch -200
  done
  ```

**Freshness check** — both preprocessing panels show a green ✓ when the processed data is current and an orange ⚠ when source files have been added or modified since the last run.  Re-run the relevant preprocessing before training.

---

### VAE + LSTM settings

#### Latent dimension
The number of values that describe a single 4-second chunk in compressed form.  Higher = richer representation, slower training, more VRAM.

| Value | When to use |
|---|---|
| 128 | Default; good for datasets under 20 k chunks |
| 256 | Recommended when you have 20 k+ chunks and ≥6 GB VRAM free |
| 64 | Only if VRAM is very tight |

Diminishing returns above 256 for the current dataset size.

#### VAE batch size
How many chunks are processed together in one gradient update.  Larger batches are more GPU-efficient but give fewer updates per epoch.

- **Too small** (< 32): GPU sits mostly idle; training is slow.
- **Too large** (> 512 with < 30 k chunks): fewer than ~20 gradient updates per epoch; the model gets fewer learning opportunities.
- **Sweet spot**: `n_train_chunks ÷ 30` rounded to a power of 2. For 28 k chunks that's ~256.

The smart defaults compute this automatically based on your chunk count and VRAM budget.

#### Epochs
One epoch = one full pass through the training data.  More is almost always better until the validation loss stops improving.

- **100 epochs**: quick check, clearly underfitting on < 30 k chunks.
- **300–500 epochs**: visible quality improvement; recommended minimum.
- **1000 epochs**: target for overnight training with < 30 k chunks.

The loss curve in the Train tab shows both training (solid) and validation (dashed) loss.  Stop when the gap between them stabilises and validation stops decreasing.

#### Learning rate (`lr_vae`)
How large each weight update is.  The cosine annealing schedule handles the decay automatically, so this is just the starting value.

- Default `1e-4` is appropriate for batch sizes of 256+.
- Use `2e-4` for smaller batches (< 128).
- Do not go above `3e-4` — training becomes unstable.

#### KL max weight
The VAE loss has two components: *reconstruction* (how accurately it rebuilds the spectrogram) and *KL divergence* (how well-structured the latent space is).  `kl_max` is the maximum weight given to the KL term.

**Posterior collapse** happens when `kl_max` is too high: the decoder learns to ignore the latent code and generate the same average output regardless of input.  Symptom: training loss looks good but all generated audio sounds identical.

- For **< 20 k chunks**: use `2e-4` (low regularisation, prioritise reconstruction)
- For **20 k+ chunks**: use `5e-4`
- Never go above `1e-3` without a large, diverse dataset

#### KL ramp fraction
The fraction of total epochs over which the KL weight linearly increases from 0 to `kl_max`.  Setting this to 0.4 means the decoder gets the first 40 % of training to learn from the latent codes before regularisation pressure kicks in.  Lowering it below 0.3 risks collapse; raising it above 0.6 wastes training capacity.

#### LSTM sequence length
The number of consecutive 4-second chunks per LSTM training example.  Each step = 4 seconds, so a sequence length of 32 covers ~128 seconds of musical context.

- **Minimum**: 16 (64 s context).  Works, but the LSTM can't learn structure longer than a minute.
- **Recommended**: 32–40 (128–160 s) — covers most of a song.  Requires at least 33 consecutive chunks per song in the dataset.
- **Maximum practical**: 64 (256 s).  Very few songs in the dataset will be long enough; most will produce no valid training sequences.

The smart defaults set this to ~50 % of the median song length in chunks.

#### LSTM batch size
LSTM training sequences are much lighter than mel chunks.  Use a larger batch (256–512) for smoother gradient estimates.

#### Number of workers
Data loading threads.  Set to half the number of physical CPU cores.  More than 4–6 rarely helps and can cause contention.

---

### RAVE settings

#### Config (`v2` vs `v1`)
- **v2** (recommended): redesigned for tonal/melodic content.  Better at piano, sustained notes, ambient textures.
- **v1**: original RAVE.  Sometimes preferred for percussive or highly rhythmic material.

#### Window size (`n_signal`)
The number of audio samples per training example.  **Must match the value used during preprocessing** — mismatches cause crashes.  The default of 131 072 corresponds to ~3 seconds at 44 100 Hz, which gives the model enough context to learn short melodic patterns.

Do not change this between preprocessing and training without re-running preprocessing.

#### Batch size
RAVE v2 at `n_signal = 131072` uses roughly 1 GB of VRAM overhead plus ~130 MB per batch item.  On an RTX 3070 with 6 GB free, batch 8 is comfortable; batch 16 is possible but cuts it close.

#### Training steps
RAVE is measured in gradient steps, not epochs.  With ~8 800 training examples and batch 8, one epoch ≈ 1 100 steps.

| Steps | Time on RTX 3070 | Quality |
|---|---|---|
| 100 k | ~1 h | Audible timbre, incoherent structure |
| 500 k | ~4–5 h | Recognisable Lambert style, some structure |
| 2 M | ~16–20 h | Full quality within hardware limits |
| 3 M | ~24 h | Diminishing returns beyond this |

The "Stop & export" button exports the best checkpoint seen so far (from periodic validation).  Stopping at 500 k and retraining from scratch with more data later is a valid strategy.

#### val_every
How often RAVE saves a checkpoint (in steps).  The default of 2 200 ≈ 2 epochs means the first real checkpoint is available after ~4 minutes, so early stops always produce a model with real weights rather than random initialisation.

---

### Generation settings

#### Temperature
Controls how far generated latent codes stray from the training distribution.

| Value | Character |
|---|---|
| 0.3–0.5 | Calm, predictable, stays close to the training data |
| 0.7–1.0 | Balanced variety |
| 1.2–2.0 | Experimental, increasingly chaotic |

For Lambert-style ambient piano, **0.4–0.6** usually sounds most coherent.

#### Griffin-Lim iterations (VAE + LSTM only)
The phase estimation algorithm that converts a mel spectrogram back to audio.  More iterations = cleaner phase, less metallic wobble.

| Iterations | Quality |
|---|---|
| 32 | Fast, very metallic |
| 64 | Bare minimum for listening |
| 128 | Good; recommended default |
| 256 | Noticeably cleaner, ~2× slower |

RAVE does not use Griffin-Lim — it decodes directly to waveform, which is why it sounds much better.

#### Duration
Both backends generate audio by sampling a latent sequence and decoding each frame.  Longer durations are proportionally more expensive but do not degrade quality.  The VAE+LSTM pipeline uses ~0.2 s per 4-second chunk; RAVE uses ~0.1 s per second of audio.

---

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

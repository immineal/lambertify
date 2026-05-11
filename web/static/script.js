/* Lambertify – full dashboard controller */

'use strict';

// ---------------------------------------------------------------------------
// Tab switching
// ---------------------------------------------------------------------------
document.querySelectorAll('.tab').forEach(btn => {
  btn.addEventListener('click', () => {
    const target = btn.dataset.tab;
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    const panel = document.getElementById(`tab-${target}`);
    if (panel) panel.classList.add('active');
    if (target === 'data')    loadDataTab();
    if (target === 'history') loadHistoryTab();
    if (target === 'models')  loadModelsTab();
  });
});

// ---------------------------------------------------------------------------
// Hardware refresh
// ---------------------------------------------------------------------------
document.getElementById('hw-refresh-btn').addEventListener('click', refreshHW);

async function refreshHW() {
  try {
    const data = await fetchJSON('/api/hw');
    data.gpus.forEach(g => updateGPUCard(g));
    updateCPUCard(data.cpu);
    if (data.gpus.length > 0) {
      window._hwLastGpuTotal = data.gpus[0].vram_total_mb;
      window._hwLastGpuFree  = data.gpus[0].vram_free_mb;
    }
  } catch (_) {}
}

function updateGPUCard(gpu) {
  const pct = gpu.vram_used_mb / gpu.vram_total_mb * 100;
  setBarPct(`vram-bar-${gpu.index}`, pct);
  setText(`vram-val-${gpu.index}`, `${gpu.vram_used_mb} / ${gpu.vram_total_mb} MB`);
  setText(`gpu-temp-${gpu.index}`, `${gpu.temperature_c}°C`);
  setText(`gpu-util-${gpu.index}`, `${gpu.utilization_pct}% util`);
  setText(`gpu-pwr-${gpu.index}`, `${Math.round(gpu.power_draw_w)}W / ${Math.round(gpu.power_limit_w)}W`);
}

function updateCPUCard(cpu) {
  const used_gb = cpu.ram_total_gb - cpu.ram_available_gb;
  const pct     = used_gb / cpu.ram_total_gb * 100;
  setBarPct('ram-bar', pct);
  setText('ram-val', `${used_gb.toFixed(1)} / ${cpu.ram_total_gb} GB`);
  setText('ram-avail', `${cpu.ram_available_gb} GB free`);
}

setInterval(refreshHW, 5000);

// ---------------------------------------------------------------------------
// Preprocess tab
// ---------------------------------------------------------------------------
// Preprocessing sliders
// ---------------------------------------------------------------------------
const hopSlider = document.getElementById('hop-fraction');
const hopVal    = document.getElementById('hop-fraction-val');

hopSlider?.addEventListener('input', () => {
  hopVal.textContent = `${parseFloat(hopSlider.value).toFixed(2)} (${Math.round(parseFloat(hopSlider.value) * 100)}% overlap)`;
});

// ---------------------------------------------------------------------------
// Data tab
// ---------------------------------------------------------------------------

let _dataTabLoaded = false;

async function loadDataTab() {
  if (!_dataTabLoaded) {
    _dataTabLoaded = true;
    await Promise.all([loadPreprocessStats(), loadDataAssessment()]);
  }
  await refreshDataStatus();
}

async function refreshDataStatus() {
  try {
    const d = await fetchJSON('/api/data_status');
    const body = document.getElementById('data-status-body');
    if (!body) return;
    const scan = d.scan || {};
    let html = `<div class="run-detail-grid">
      <div class="run-detail-kv"><span class="run-detail-key">Folder</span><span class="run-detail-val">${d.data_dir}</span></div>
      <div class="run-detail-kv"><span class="run-detail-key">Audio files</span><span class="run-detail-val">${scan.n_files ?? '?'}</span></div>
      <div class="run-detail-kv"><span class="run-detail-key">Total size</span><span class="run-detail-val">${scan.total_bytes ? (scan.total_bytes/1e9).toFixed(2)+' GB' : '—'}</span></div>
      <div class="run-detail-kv"><span class="run-detail-key">Newest file</span><span class="run-detail-val">${scan.newest_mtime_str ?? '—'}</span></div>
    </div>`;
    if (d.vae_stale) {
      html += `<div class="warn-item" style="margin-top:8px">⚠ VAE stale: ${d.vae_stale_reason}</div>`;
    } else {
      const snap = d.vae_snapshot;
      html += `<div style="color:var(--ok);font-size:.82rem;margin-top:4px">✓ VAE preprocessing current (${snap?.n_files ?? '?'} files, ${snap?.preprocessed_at ?? '?'})</div>`;
    }
    if (d.rave_stale) {
      html += `<div class="warn-item" style="margin-top:4px">⚠ RAVE stale: ${d.rave_stale_reason}</div>`;
    } else if (d.rave_snapshot) {
      const snap = d.rave_snapshot;
      html += `<div style="color:var(--ok);font-size:.82rem;margin-top:4px">✓ RAVE preprocessing current (${snap?.n_files ?? '?'} files, ${snap?.preprocessed_at ?? '?'})</div>`;
    }
    body.innerHTML = html;
    // Update RAVE status in Train tab too
    const statusHtml = d.rave_stale
      ? `<span style="color:var(--warn)">⚠ Re-preprocess needed — go to Data tab</span>`
      : '<span style="color:var(--ok)">✓ RAVE data ready</span>';
    for (const id of ['rave-preprocess-status-data', 'rave-preprocess-status']) {
      const el = document.getElementById(id);
      if (el) el.innerHTML = statusHtml;
    }
  } catch (_) {}
}

document.getElementById('data-dir-save-btn')?.addEventListener('click', async () => {
  const input  = document.getElementById('data-dir-input');
  const status = document.getElementById('data-dir-status');
  if (!input) return;
  try {
    await postJSON('/api/config', { data_dir: input.value.trim() });
    if (status) { status.textContent = '✓ Saved'; status.style.color = 'var(--ok)'; }
    _dataTabLoaded = false;
    await loadDataTab();
    setTimeout(() => { if (status) status.textContent = ''; }, 3000);
  } catch (e) {
    if (status) { status.textContent = 'Error: ' + e.message; status.style.color = 'var(--danger)'; }
  }
});

document.getElementById('data-dir-input')?.addEventListener('keydown', e => {
  if (e.key === 'Enter') document.getElementById('data-dir-save-btn')?.click();
});

async function loadPreprocessStats() {
  try {
    const d = await fetchJSON('/api/preprocess_stats');
    if (d.ready) showPreprocessStats(d);
  } catch (_) {}
}

function showPreprocessStats(d) {
  const box = document.getElementById('preprocess-stats');
  if (!box) return;
  box.innerHTML = `
    <div class="stat-item"><div class="stat-key">Chunks</div><div class="stat-val">${d.n_chunks?.toLocaleString() ?? '—'}</div></div>
    <div class="stat-item"><div class="stat-key">Songs</div><div class="stat-val">${d.n_songs ?? '—'}</div></div>
    <div class="stat-item"><div class="stat-key">Shape</div><div class="stat-val">${d.shape?.join(' × ') ?? '—'}</div></div>
    <div class="stat-item"><div class="stat-key">Chunk dur.</div><div class="stat-val">4 s (${d.frames_per_chunk} frames)</div></div>
    <div class="stat-item"><div class="stat-key">Mel bins</div><div class="stat-val">${d.n_mels ?? 128}</div></div>
    <div class="stat-item"><div class="stat-key">Dataset</div><div class="stat-val">${d.shape ? (d.shape.reduce((a,b)=>a*b,4)/1e9).toFixed(2)+' GB' : '—'}</div></div>
  `;
  box.classList.remove('hidden');
}

async function loadDataAssessment() {
  const el = document.getElementById('data-assessment');
  if (!el) return;
  try {
    const d = await fetchJSON('/api/assess_data');
    if (!d.n_chunks) return;
    el.classList.remove('hidden');
    const tierColor = { insufficient: 'var(--danger)', marginal: 'var(--warn)',
                        decent: '#e8c33a', good: 'var(--ok)', excellent: 'var(--ok)' };
    const col = tierColor[d.tier] || 'var(--text-dim)';
    let html = `<div class="param-group">
      <div class="param-group-title" style="color:${col}">Data assessment — ${d.tier.toUpperCase()}</div>
      <p class="hint" style="color:${col};font-weight:600">${d.summary}</p>
      <div class="hint" style="margin-top:4px">${d.n_chunks.toLocaleString()} chunks · ${d.n_songs} songs · ${d.hours_of_audio}h audio</div>
      <div style="margin-top:10px;display:flex;flex-direction:column;gap:8px">`;
    d.actions.forEach(a => {
      const pc = a.priority.startsWith('HIGH') ? 'var(--warn)'
               : a.priority.startsWith('MEDIUM') ? 'var(--accent-h)' : 'var(--text-muted)';
      html += `<div style="border-left:3px solid ${pc};padding-left:10px">
        <div style="font-size:.78rem;font-weight:700;color:${pc};text-transform:uppercase">${a.priority}</div>
        <div style="font-size:.85rem;font-weight:600;color:var(--text);margin:2px 0">${a.action}</div>
        <div class="hint" style="white-space:pre-wrap">${a.effect}</div>
        <div style="font-size:.72rem;color:var(--text-muted);margin-top:2px">Cost: ${a.cost}</div>
      </div>`;
    });
    html += '</div></div>';
    el.innerHTML = html;
  } catch (_) {}
}

function setPreprocessStatus(msg, color) {
  const el = document.getElementById('preprocess-status-line');
  if (el) { el.textContent = msg; el.style.color = color || 'var(--text-dim)'; }
}

// VAE preprocessing
document.getElementById('preprocess-btn')?.addEventListener('click', async () => {
  const btn = document.getElementById('preprocess-btn');
  btn.disabled = true;
  setPreprocessStatus('VAE preprocessing starting…', 'var(--warn)');
  try {
    await postJSON('/api/preprocess', { hop_fraction: parseFloat(hopSlider?.value || 0.2) });
    pollVAEPreprocess(btn);
  } catch (e) {
    setPreprocessStatus('Error: ' + e.message, 'var(--danger)');
    btn.disabled = false;
  }
});

function pollVAEPreprocess(btn) {
  const iv = setInterval(async () => {
    try {
      const s     = await fetchJSON('/api/train_status');
      const lines = await fetchJSON('/api/train_log?n=100');
      const inner = document.getElementById('preprocess-log-inner');
      if (inner) {
        inner.textContent = lines.lines.slice(-80).join('\n');
        inner.parentElement.scrollTop = inner.parentElement.scrollHeight;
      }
      if (s.status !== 'preprocessing') {
        clearInterval(iv);
        btn.disabled = false;
        if (s.status === 'idle') {
          setPreprocessStatus('VAE preprocessing complete', 'var(--ok)');
          await Promise.all([loadPreprocessStats(), loadDataAssessment()]);
          _dataTabLoaded = false;
          await refreshDataStatus();
        } else {
          setPreprocessStatus('VAE preprocessing failed — see log', 'var(--danger)');
        }
      } else {
        setPreprocessStatus('VAE preprocessing in progress…', 'var(--warn)');
      }
    } catch (_) {}
  }, 1500);
}

// RAVE preprocessing (button in Data tab)
document.getElementById('rave-preprocess-btn')?.addEventListener('click', async () => {
  const btn = document.getElementById('rave-preprocess-btn');
  btn.disabled = true;
  btn.textContent = 'Preprocessing…';
  setPreprocessStatus('RAVE preprocessing starting…', 'var(--warn)');
  try {
    const sr = intVal('rave-sr', 44100);
    await postJSON('/api/rave/preprocess', { sample_rate: sr });
    pollRaveInDataTab(btn);
  } catch (e) {
    setPreprocessStatus('RAVE failed: ' + e.message, 'var(--danger)');
    btn.disabled = false;
    btn.textContent = 'Run RAVE preprocessing';
  }
});

function pollRaveInDataTab(btn) {
  const iv = setInterval(async () => {
    try {
      const s = await fetchJSON('/api/rave/status');
      const inner = document.getElementById('preprocess-log-inner');
      if (inner && s.log_tail?.length) {
        inner.textContent = s.log_tail.slice(-80).join('\n');
        inner.parentElement.scrollTop = inner.parentElement.scrollHeight;
      }
      if (s.status !== 'preprocessing') {
        clearInterval(iv);
        btn.disabled = false;
        btn.textContent = 'Run RAVE preprocessing';
        if (s.status === 'idle') {
          setPreprocessStatus('RAVE preprocessing complete', 'var(--ok)');
          _dataTabLoaded = false;
          await refreshDataStatus();
        } else if (s.status === 'error') {
          setPreprocessStatus('RAVE failed — see log', 'var(--danger)');
        }
      } else {
        setPreprocessStatus('RAVE preprocessing in progress…', 'var(--warn)');
      }
    } catch (_) {}
  }, 1500);
}

// ---------------------------------------------------------------------------
// Train tab — smart defaults
// ---------------------------------------------------------------------------
document.getElementById('smart-defaults-btn').addEventListener('click', applySmartDefaults);
document.getElementById('vram-refresh-btn').addEventListener('click', updateVRAMEstimate);

// Auto-recalculate VRAM estimate on param changes
['p-vae-batch','p-lstm-batch','p-latent-dim','p-seq-len'].forEach(id => {
  const el = document.getElementById(id);
  if (el) el.addEventListener('change', updateVRAMEstimate);
});

async function applySmartDefaults() {
  const btn = document.getElementById('smart-defaults-btn');
  btn.disabled = true;
  try {
    const rec = await fetchJSON('/api/params');
    setInputVal('p-vae-batch',   rec.vae_batch_size);
    setInputVal('p-lstm-batch',  rec.lstm_batch_size);
    setInputVal('p-vae-epochs',  rec.vae_epochs);
    setInputVal('p-lstm-epochs', rec.lstm_epochs);
    setInputVal('p-lr-vae',      rec.lr_vae);
    setInputVal('p-lr-lstm',     rec.lr_lstm);
    setInputVal('p-kl-max',      rec.kl_max);
    setInputVal('p-seq-len',     rec.seq_len);
    setInputVal('p-workers',          rec.num_workers);
    setInputVal('p-kl-ramp',          rec.kl_ramp_pct ?? 0.4);
    setInputVal('p-checkpoint-every', rec.checkpoint_every ?? 50);

    const latentSel = document.getElementById('p-latent-dim');
    const opts = Array.from(latentSel.options).map(o => parseInt(o.value));
    const closest = opts.reduce((a, b) => Math.abs(b - rec.latent_dim) < Math.abs(a - rec.latent_dim) ? b : a);
    latentSel.value = closest;

    if (rec.device && document.getElementById('p-device')) {
      const devSel = document.getElementById('p-device');
      const existing = Array.from(devSel.options).find(o => o.value === rec.device);
      if (existing) devSel.value = rec.device;
    }

    // Show warnings
    if (rec.warnings?.length) {
      showWarnList('vram-est-warnings', rec.warnings, 'warn');
    }

    // Show the full rationale in a collapsible section
    if (rec.rationale && typeof rec.rationale === 'object') {
      const rationaleEl = document.getElementById('smart-rationale');
      const rationaleWrap = document.getElementById('smart-rationale-wrap');
      if (rationaleEl && rationaleWrap) {
        const entries = Object.entries(rec.rationale);
        rationaleEl.innerHTML = entries.map(([k, v]) =>
          `<div class="run-detail-kv" style="margin-bottom:6px;flex-wrap:wrap">
            <span class="run-detail-key" style="min-width:120px;font-weight:600">${k}</span>
            <span class="run-detail-val" style="font-family:inherit;color:var(--text-dim);flex:1;min-width:200px">${v}</span>
          </div>`
        ).join('');
        rationaleWrap.classList.remove('hidden');
      }
    }

    await updateVRAMEstimate();
  } finally {
    btn.disabled = false;
  }
}

async function updateVRAMEstimate() {
  const params = gatherTrainParams();
  try {
    const est = await postJSON('/api/estimate_vram', {
      vae_batch_size:  params.vae_batch_size,
      lstm_batch_size: params.lstm_batch_size,
      latent_dim:      params.latent_dim,
      seq_len:         params.seq_len,
    });

    const freeM = est.vram_free_mb || window._hwLastGpuFree || 8192;

    setEstBar('vae-est-bar',  'vae-est-val',  est.vae_mb,  freeM);
    setEstBar('lstm-est-bar', 'lstm-est-val', est.lstm_mb, freeM);
    showWarnList('vram-est-warnings', est.warnings || [], 'err');
  } catch (_) {}
}

function setEstBar(barId, valId, mb, freeM) {
  const pct = Math.min(100, mb / freeM * 100);
  const bar = document.getElementById(barId);
  const val = document.getElementById(valId);
  if (bar) {
    bar.style.width = pct + '%';
    bar.className = 'bar-fill ' + (pct > 90 ? 'bar-danger' : pct > 70 ? 'bar-warn' : '');
  }
  if (val) val.textContent = mb + ' MB';
}

// ---------------------------------------------------------------------------
// Train tab — start / stop training
// ---------------------------------------------------------------------------
document.getElementById('train-btn').addEventListener('click', startTraining);
document.getElementById('stop-btn').addEventListener('click', stopTraining);

let _trainPollInterval = null;
let _lossChart = null;

function initLossChart() {
  const ctx = document.getElementById('loss-chart').getContext('2d');
  if (_lossChart) { _lossChart.destroy(); }
  _lossChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        { label: 'VAE train',  data: [], borderColor: '#7b6ee8', backgroundColor: 'rgba(123,110,232,.08)', tension: .3, pointRadius: 0, borderWidth: 2, yAxisID: 'yVAE', fill: true },
        { label: 'VAE val',    data: [], borderColor: '#9b8ef8', tension: .3, pointRadius: 0, borderWidth: 1.5, borderDash: [5,4], yAxisID: 'yVAE' },
        { label: 'LSTM train', data: [], borderColor: '#4caf7d', backgroundColor: 'rgba(76,175,125,.06)', tension: .3, pointRadius: 0, borderWidth: 2, yAxisID: 'yLSTM', fill: true },
        { label: 'LSTM val',   data: [], borderColor: '#7de0ae', tension: .3, pointRadius: 0, borderWidth: 1.5, borderDash: [5,4], yAxisID: 'yLSTM' },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      animation: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { labels: { color: '#7a79a0', font: { size: 11 }, boxWidth: 14 } },
        title: { display: false },
      },
      scales: {
        x: {
          ticks: { color: '#4a4a68', maxTicksLimit: 12 },
          grid:  { color: '#1e1e28' },
          title: { display: true, text: 'Epoch', color: '#4a4a68', font: { size: 10 } },
        },
        yVAE: {
          type: 'linear', position: 'left',
          ticks: { color: '#7b6ee8', font: { size: 10 } },
          grid:  { color: '#1e1e28' },
          title: { display: true, text: 'VAE loss', color: '#7b6ee8', font: { size: 10 } },
        },
        yLSTM: {
          type: 'linear', position: 'right',
          ticks: { color: '#4caf7d', font: { size: 10 } },
          grid:  { drawOnChartArea: false },
          title: { display: true, text: 'LSTM loss', color: '#4caf7d', font: { size: 10 } },
        },
      },
    },
  });
}

async function startTraining() {
  const btn  = document.getElementById('train-btn');
  const stop = document.getElementById('stop-btn');
  btn.disabled = true;
  stop.classList.remove('hidden');

  const params = gatherTrainParams();
  try {
    await postJSON('/api/train', params);
    initLossChart();
    startTrainPoll();
  } catch (e) {
    alert('Failed to start training: ' + e.message);
    btn.disabled = false;
    stop.classList.add('hidden');
  }
}

async function stopTraining() {
  await postJSON('/api/train_stop', {});
  setText('live-status', 'Stop requested — finishing current epoch…');
}

function startTrainPoll() {
  if (_trainPollInterval) clearInterval(_trainPollInterval);
  _trainPollInterval = setInterval(pollTrainStatus, 1500);
}

async function pollTrainStatus() {
  try {
    const s = await fetchJSON('/api/train_status');
    const logData = await fetchJSON('/api/train_log?n=200');

    updateTrainUI(s);
    updateTrainLog(logData.lines);

    if (s.status === 'done' || s.status === 'error' || s.status === 'idle') {
      clearInterval(_trainPollInterval);
      _trainPollInterval = null;
      document.getElementById('train-btn').disabled = false;
      document.getElementById('stop-btn').classList.add('hidden');
      updateStatusBanner(s);
      refreshHW();
    }
  } catch (_) {}
}

function updateTrainUI(s) {
  // Status string
  // VAE training progress bar
  const progWrap  = document.getElementById('train-progress-wrap');
  const progBar   = document.getElementById('train-progress-bar');
  const progVal   = document.getElementById('train-progress-val');
  const progLabel = document.getElementById('progress-label');
  if (s.status === 'training' && s.total_epochs > 0) {
    if (progWrap)  progWrap.style.display = '';
    if (progLabel) progLabel.textContent  = 'Epochs';
    const pct = Math.min(100, s.epoch / s.total_epochs * 100);
    if (progBar) progBar.style.width = pct.toFixed(1) + '%';
    if (progVal) progVal.textContent  = `${s.epoch} / ${s.total_epochs}`;
  } else if (progWrap && s.status !== 'training') {
    progWrap.style.display = 'none';
  }

  let statusStr = 'Idle';
  if (s.status === 'training') {
    statusStr = `Training ${s.stage?.toUpperCase() || ''} — epoch ${s.epoch}/${s.total_epochs}`;
  } else if (s.status === 'preprocessing') {
    statusStr = 'Preprocessing…';
  } else if (s.status === 'done') {
    statusStr = `Done (${s.finished || ''})`;
  } else if (s.status === 'error') {
    statusStr = 'Error — see log below';
  }
  const liveEl = document.getElementById('live-status');
  liveEl.textContent = statusStr;
  liveEl.style.color = s.status === 'error' ? 'var(--danger)' : s.status === 'done' ? 'var(--ok)' : 'var(--text-dim)';

  // Live VRAM gauge (reads real total from last hw snapshot)
  const vramBar   = document.getElementById('live-vram-bar');
  const vramVal   = document.getElementById('live-vram-val');
  const vramTotal = window._hwLastGpuTotal || 8192;
  if (s.vram_mb > 0) {
    const pct = Math.min(100, s.vram_mb / vramTotal * 100);
    if (vramBar) {
      vramBar.style.width = pct + '%';
      vramBar.className = 'bar-fill ' + (pct > 90 ? 'bar-danger' : pct > 75 ? 'bar-warn' : '');
    }
    if (vramVal) vramVal.textContent = `${s.vram_mb} / ${vramTotal} MB  (${pct.toFixed(0)}%)`;
  }

  // Loss chart
  if (_lossChart && s.history) {
    const h = s.history;
    const maxLen = Math.max(
      h.vae_train?.length || 0,
      h.lstm_train?.length || 0,
    );
    _lossChart.data.labels = Array.from({ length: maxLen }, (_, i) => i + 1);
    _lossChart.data.datasets[0].data = h.vae_train  || [];
    _lossChart.data.datasets[1].data = h.vae_val    || [];
    _lossChart.data.datasets[2].data = h.lstm_train || [];
    _lossChart.data.datasets[3].data = h.lstm_val   || [];
    _lossChart.update('none');
  }
}

function updateTrainLog(lines) {
  const inner = document.getElementById('train-log-inner');
  if (!inner) return;
  // Show last 100 lines only — keeps the DOM lean
  inner.textContent = lines.slice(-100).join('\n');
  const box = document.getElementById('train-log-box');
  if (box) box.scrollTop = box.scrollHeight;
}

async function updateStatusBanner(s) {
  const banner = document.getElementById('status-banner');

  // Fetch fresh model state from server (the banner shows active model name)
  try {
    const ms = await fetchJSON('/api/models');
    const am = ms.models.find(m => m.id === ms.active_id);
    if (banner && am && s.vae_ready) {
      let text = `Active: ${am.name}`;
      if (am.backend === 'vae_lstm') {
        text += ` — VAE ep${am.vae_epoch || '?'}`;
        if (am.vae_val_loss) text += ` (val=${am.vae_val_loss.toFixed(4)})`;
        if (!am.lstm_path)   text += ' · no LSTM';
      } else if (am.backend === 'rave') {
        text += ` — RAVE ${am.config?.config || ''}`;
      }
      if (ms.cache?.loaded) text += ' · loaded in memory';
      banner.innerHTML = text;
      banner.className = 'banner banner-ok';
    } else if (banner && !s.vae_ready) {
      banner.textContent = 'No model trained yet — preprocess data then start training.';
      banner.className = 'banner banner-warn';
    }
  } catch (_) {}

  // Determine if active model is RAVE
  const isRave = am?.backend === 'rave';

  // Generate tab warnings + controls
  const noVae     = document.getElementById('gen-no-vae-warn');
  const noLstm    = document.getElementById('gen-no-lstm-warn');
  const vaeCtrls  = document.getElementById('gen-vae-controls');
  if (noVae)    noVae.style.display    = s.vae_ready ? 'none' : '';
  if (noLstm)   noLstm.style.display   = (s.vae_ready && !s.lstm_ready && !isRave) ? '' : 'none';
  if (vaeCtrls) vaeCtrls.style.display = isRave ? 'none' : '';

  // Generate button
  const genBtn = document.getElementById('generate-btn');
  if (genBtn && s.vae_ready) genBtn.disabled = false;
}

// ---------------------------------------------------------------------------
// Generate tab
// ---------------------------------------------------------------------------
const gDuration    = document.getElementById('g-duration');
const gTemp        = document.getElementById('g-temperature');
const gGlIters     = document.getElementById('g-gl-iters');

gDuration.addEventListener('input', () =>
  setText('g-duration-val', `${gDuration.value} s`));
gTemp.addEventListener('input', () =>
  setText('g-temperature-val', parseFloat(gTemp.value).toFixed(2)));
gGlIters.addEventListener('input', () =>
  setText('g-gl-iters-val', gGlIters.value));

document.getElementById('generate-btn')?.addEventListener('click', async () => {
  const btn = document.getElementById('generate-btn');
  btn.disabled = true;
  document.getElementById('player-section').classList.add('hidden');
  document.getElementById('gen-progress').classList.remove('hidden');
  setText('gen-progress-msg', 'Generating…');

  const seedVal = document.getElementById('g-seed')?.value?.trim();

  try {
    const res = await postJSON('/api/generate', {
      duration:          parseFloat(gDuration.value),
      temperature:       parseFloat(gTemp.value),
      griffin_lim_iters: parseInt(gGlIters.value),
      seed:              seedVal || null,
    });
    await pollGenJob(res.job_id);
  } catch (e) {
    document.getElementById('gen-progress').classList.add('hidden');
    alert('Generation failed: ' + e.message);
  } finally {
    btn.disabled = false;
  }
});

const _genMsgs = [
  'Sampling latent codes…',
  'Decoding mel spectrograms…',
  'Running Griffin-Lim vocoder…',
  'Crossfading chunks…',
  'Almost there…',
];

async function pollGenJob(jobId) {
  let tick = 0;
  return new Promise((resolve, reject) => {
    const iv = setInterval(async () => {
      try {
        const d = await fetchJSON(`/api/job/${jobId}`);
        setText('gen-progress-msg', _genMsgs[tick % _genMsgs.length]);
        tick++;
        if (d.status === 'done') {
          clearInterval(iv);
          document.getElementById('gen-progress').classList.add('hidden');
          await loadGeneratedAudio(jobId);
          resolve();
        } else if (d.status === 'error') {
          clearInterval(iv);
          document.getElementById('gen-progress').classList.add('hidden');
          reject(new Error(d.error?.split('\n')[0] || 'Generation error'));
        }
      } catch (e) { clearInterval(iv); reject(e); }
    }, 1500);
  });
}

async function loadGeneratedAudio(jobId) {
  const url = `/api/audio/${jobId}`;
  const player = document.getElementById('audio-player');
  const dl     = document.getElementById('download-link');
  player.src = url;
  dl.href    = url;
  document.getElementById('player-section').classList.remove('hidden');
  player.addEventListener('canplaythrough', () => drawWaveform(url), { once: true });
}

async function drawWaveform(url) {
  const canvas = document.getElementById('waveform');
  const ctx    = canvas.getContext('2d');
  canvas.width  = canvas.offsetWidth;
  canvas.height = 80;

  try {
    const resp   = await fetch(url);
    const buf    = await resp.arrayBuffer();
    const ac     = new AudioContext();
    const decoded = await ac.decodeAudioData(buf);
    const data   = decoded.getChannelData(0);
    const step   = Math.ceil(data.length / canvas.width);
    const h2     = canvas.height / 2;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = '#7b6ee8';
    ctx.lineWidth   = 1;
    ctx.beginPath();
    for (let x = 0; x < canvas.width; x++) {
      const slice = data.slice(x * step, (x + 1) * step);
      const max   = Math.max(...slice.map(Math.abs));
      ctx.moveTo(x, h2 - max * h2 * 0.95);
      ctx.lineTo(x, h2 + max * h2 * 0.95);
    }
    ctx.stroke();
    await ac.close();
  } catch (_) {}
}

// ---------------------------------------------------------------------------
// Outputs tab
// ---------------------------------------------------------------------------
async function loadOutputsList() {
  const list = document.getElementById('outputs-list');
  list.textContent = 'Loading…';
  try {
    const files = await fetchJSON('/api/outputs');
    if (!files.length) {
      list.textContent = 'No outputs yet. Generate something first.';
      list.style.color = 'var(--text-dim)';
      return;
    }
    list.innerHTML = '';
    files.forEach(f => {
      const div = document.createElement('div');
      div.className = 'output-item';
      const ts = new Date(f.ts * 1000).toLocaleString();
      div.innerHTML = `
        <span class="output-name">${f.name}</span>
        <span class="output-size">${f.size_mb} MB</span>
        <span class="output-size" style="min-width:130px;color:var(--text-muted)">${ts}</span>
        <audio controls src="${f.url}" class="output-play" style="flex:2;min-width:180px;max-width:280px"></audio>
        <a href="${f.url}" download="${f.name}" class="btn-secondary output-play" style="padding:5px 12px;font-size:.78rem;text-decoration:none">↓</a>
      `;
      list.appendChild(div);
    });
  } catch (e) {
    list.textContent = 'Failed to load outputs: ' + e.message;
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function gatherTrainParams() {
  const stage = document.querySelector('input[name="stage"]:checked')?.value || 'both';
  return {
    stage,
    vae_batch_size:    intVal('p-vae-batch',       32),
    lstm_batch_size:   intVal('p-lstm-batch',      128),
    latent_dim:        intVal('p-latent-dim',      128),
    vae_epochs:        intVal('p-vae-epochs',       50),
    lstm_epochs:       intVal('p-lstm-epochs',      30),
    lr_vae:            floatVal('p-lr-vae',        2e-4),
    lr_lstm:           floatVal('p-lr-lstm',       1e-3),
    kl_max:            floatVal('p-kl-max',        5e-4),
    kl_ramp_pct:       floatVal('p-kl-ramp',       0.4),
    seq_len:           intVal('p-seq-len',          16),
    num_workers:       intVal('p-workers',           2),
    checkpoint_every:  intVal('p-checkpoint-every', 50),
    device:            document.getElementById('p-device')?.value || 'cuda',
  };
}

function showWarnList(containerId, warnings, cls) {
  const container = document.getElementById(containerId);
  if (!container) return;
  container.innerHTML = '';
  warnings.forEach(w => {
    const d = document.createElement('div');
    d.className = `warn-item ${cls || ''}`;
    d.textContent = w;
    container.appendChild(d);
  });
}

async function fetchJSON(url) {
  const r = await fetch(url);
  if (!r.ok) { const e = await r.json().catch(() => ({})); throw new Error(e.error || `HTTP ${r.status}`); }
  return r.json();
}

async function postJSON(url, body) {
  const r = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!r.ok) { const e = await r.json().catch(() => ({})); throw new Error(e.error || `HTTP ${r.status}`); }
  return r.json();
}

function setText(id, val) { const el = document.getElementById(id); if (el) el.textContent = val; }
function setInputVal(id, val) { const el = document.getElementById(id); if (el) el.value = val; }
function intVal(id, def) { const el = document.getElementById(id); return el ? (parseInt(el.value) || def) : def; }
function floatVal(id, def) { const el = document.getElementById(id); return el ? (parseFloat(el.value) || def) : def; }

function setBarPct(id, pct) {
  const bar = document.getElementById(id);
  if (!bar) return;
  bar.style.width = Math.min(100, pct).toFixed(1) + '%';
  bar.className = 'bar-fill ' + (pct > 90 ? 'bar-danger' : pct > 75 ? 'bar-warn' : pct > 50 ? '' : 'bar-ok');
}

// ---------------------------------------------------------------------------
// Models tab
// ---------------------------------------------------------------------------

document.getElementById('models-refresh-btn')?.addEventListener('click', loadModelsTab);

async function loadModelsTab() {
  const list   = document.getElementById('models-list');
  const status = document.getElementById('model-cache-status');
  try {
    const d = await fetchJSON('/api/models');
    const cache = d.cache;

    // Cache status bar
    if (status) {
      if (cache.loaded) {
        status.textContent = `Loaded in memory: "${d.models.find(m=>m.id===cache.model_id)?.name || cache.model_id}" on ${cache.device} — ${cache.vram_mb} MB VRAM`;
        status.style.color = 'var(--ok)';
      } else {
        status.textContent = 'No model loaded in memory (will load lazily on generate)';
        status.style.color = 'var(--text-dim)';
      }
    }

    if (!d.models.length) {
      list.innerHTML = '<p class="hint">No models registered yet. Train something first.</p>';
      return;
    }

    list.innerHTML = '';
    d.models.forEach(m => {
      const card = document.createElement('div');
      card.className = 'model-card';
      const isActive = m.id === d.active_id;
      const isLoaded = cache.loaded && cache.model_id === m.id;

      const backendTag = m.backend === 'rave'
        ? '<span class="chip" style="background:rgba(76,175,125,.15);color:var(--ok);border-color:rgba(76,175,125,.3)">RAVE</span>'
        : '<span class="chip">VAE+LSTM</span>';

      const qualityInfo = m.backend === 'vae_lstm'
        ? `ep ${m.vae_epoch ?? '?'} · val ${m.vae_val_loss?.toFixed(5) ?? '?'} · latent ${m.config?.latent_dim ?? '?'}`
        : `${m.config?.config ?? ''} · ${(m.steps/1000)?.toFixed(0) ?? '?'}k steps`;

      card.innerHTML = `
        <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap">
          <div style="flex:1;min-width:200px">
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">
              ${backendTag}
              ${isActive ? '<span class="chip" style="background:rgba(123,110,232,.2);color:var(--accent-h);border-color:rgba(123,110,232,.4)">ACTIVE</span>' : ''}
              ${isLoaded ? '<span class="chip" style="background:rgba(76,175,125,.15);color:var(--ok)">IN MEMORY</span>' : ''}
              <span class="model-name" id="name-${m.id}" style="font-weight:600;color:var(--text)">${m.name}</span>
            </div>
            <div class="hint">${qualityInfo} · ${m.created}</div>
            ${m.checkpoints?.length ? `<div class="hint" style="margin-top:2px">${m.checkpoints.length} periodic checkpoints</div>` : ''}
          </div>
          <div style="display:flex;gap:6px;flex-wrap:wrap">
            ${!isActive ? `<button class="btn-text" onclick="activateModel('${m.id}')">Set active</button>` : ''}
            ${!isLoaded ? `<button class="btn-secondary" style="padding:5px 12px;font-size:.78rem" onclick="loadModel('${m.id}')">Load</button>` : ''}
            ${isLoaded  ? `<button class="btn-danger"    style="padding:5px 12px;font-size:.78rem" onclick="unloadModel()">Unload</button>` : ''}
            <button class="btn-secondary" style="padding:5px 12px;font-size:.78rem" onclick="renameModel('${m.id}', '${m.name.replace(/'/g, "\\'")}')">Rename</button>
            <button class="btn-danger"    style="padding:5px 12px;font-size:.78rem" onclick="deleteModel('${m.id}')">Delete</button>
          </div>
        </div>
        ${m.checkpoints?.length ? buildCheckpointList(m) : ''}
      `;
      list.appendChild(card);
    });
  } catch (e) {
    if (list) list.textContent = 'Failed to load models: ' + e.message;
  }
}

function buildCheckpointList(m) {
  if (!m.checkpoints?.length) return '';
  const items = m.checkpoints
    .sort((a, b) => a.epoch - b.epoch)
    .map(c => `<span class="chip" style="cursor:pointer" title="${c.path}" onclick="activateCheckpoint('${m.id}', '${c.path}', ${c.epoch})">ep ${c.epoch}</span>`)
    .join('');
  return `<div style="margin-top:8px;display:flex;flex-wrap:wrap;gap:6px;padding-top:8px;border-top:1px solid var(--border)">
    <span class="hint" style="align-self:center">Checkpoints:</span> ${items}
  </div>`;
}

async function activateModel(modelId) {
  await postJSON('/api/models/activate', { model_id: modelId });
  loadModelsTab();
}

async function loadModel(modelId) {
  const btn = event.target;
  btn.disabled = true;
  btn.textContent = 'Loading…';
  try {
    const r = await postJSON('/api/models/load', { model_id: modelId });
    loadModelsTab();
    // Refresh cache status in generate tab too
    updateStatusBanner({ vae_ready: true, lstm_ready: true });
  } catch (e) {
    alert('Load failed: ' + e.message);
  } finally {
    btn.disabled = false;
  }
}

async function unloadModel() {
  await postJSON('/api/models/unload', {});
  loadModelsTab();
}

async function renameModel(modelId, currentName) {
  const newName = prompt('New name:', currentName);
  if (newName && newName !== currentName) {
    await postJSON('/api/models/rename', { model_id: modelId, name: newName });
    loadModelsTab();
  }
}

async function deleteModel(modelId) {
  const delFiles = confirm(
    'Delete model files from disk too?\n(Cancel = remove from registry only)'
  );
  if (!confirm('Are you sure you want to delete this model?')) return;
  await postJSON('/api/models/delete', { model_id: modelId, delete_files: delFiles });
  loadModelsTab();
}

async function activateCheckpoint(modelId, ckptPath, epoch) {
  // Load this specific checkpoint as the active VAE
  await postJSON('/api/models/load', { model_id: modelId, vae_override: ckptPath });
  alert(`Loaded checkpoint ep${epoch}`);
  loadModelsTab();
}


// ---------------------------------------------------------------------------
// Backend selector
// ---------------------------------------------------------------------------

let _activeBackend = 'vae_lstm';

function selectBackend(backend) {
  _activeBackend = backend;
  document.getElementById('vae-panel').style.display  = backend === 'vae_lstm' ? '' : 'none';
  document.getElementById('rave-panel').style.display = backend === 'rave'     ? '' : 'none';
  document.getElementById('backend-vae-btn').classList.toggle('active', backend === 'vae_lstm');
  document.getElementById('backend-rave-btn').classList.toggle('active', backend === 'rave');

  // Right column: hide/show VAE loss chart for RAVE (it doesn't output a loss curve)
  const chartWrap = document.getElementById('vae-chart-wrap');
  if (chartWrap) chartWrap.style.display = backend === 'vae_lstm' ? '' : 'none';

  if (backend === 'rave') {
    refreshRaveUI();
  }
}


// ---------------------------------------------------------------------------
// RAVE backend
// ---------------------------------------------------------------------------

let _ravePollIv = null;

// Wire buttons
document.getElementById('rave-install-btn')?.addEventListener('click', async () => {
  const btn = document.getElementById('rave-install-btn');
  btn.disabled = true;
  btn.textContent = 'Installing…';
  showRaveProgress('installing');
  try {
    await postJSON('/api/rave/install', {});
    startRavePoll();
  } catch (e) {
    btn.disabled = false;
    btn.textContent = 'Install acids-rave';
  }
});

document.getElementById('rave-preprocess-btn')?.addEventListener('click', async () => {
  const btn = document.getElementById('rave-preprocess-btn');
  btn.disabled = true;
  btn.textContent = 'Preprocessing…';

  showRaveProgress('preprocessing');
  try {
    const sr = intVal('rave-sr', 44100);
    await postJSON('/api/rave/preprocess', { sample_rate: sr });
    startRavePoll();
  } catch (e) {
  
    btn.disabled = false;
    btn.textContent = 'Run RAVE preprocess';
  }
});

document.getElementById('rave-smart-btn')?.addEventListener('click', async () => {
  try {
    const p = await fetchJSON('/api/rave/params');
    if (p.config)      setInputVal('rave-config',   p.config);
    if (p.n_signal)    setInputVal('rave-n-signal',  p.n_signal);
    if (p.batch_size)  setInputVal('rave-batch',     p.batch_size);
    if (p.n_steps)     setInputVal('rave-steps',     p.n_steps);
    if (p.workers)     setInputVal('rave-workers',   p.workers);
    const rat = document.getElementById('rave-rationale');
    if (rat && p.rationale) {
      rat.textContent = Object.entries(p.rationale).map(([k,v]) => `${k}: ${v}`).join('\n');
    }
    if (p.warnings?.length) {
      const w = document.getElementById('rave-rationale');
      if (w) w.textContent += '\n\n⚠ ' + p.warnings.join('\n⚠ ');
    }
  } catch (_) {}
});

document.getElementById('rave-train-btn')?.addEventListener('click', async () => {
  const btn  = document.getElementById('rave-train-btn');
  const stop = document.getElementById('rave-stop-btn');
  btn.disabled = true;
  stop.classList.remove('hidden');
  showRaveProgress('training');
  // Persist RAVE settings
  await postJSON('/api/config', { rave: {
    name:       document.getElementById('rave-name')?.value  || 'lambert',
    config:     document.getElementById('rave-config')?.value || 'v2',
    batch_size: intVal('rave-batch', 8),
    n_steps:    intVal('rave-steps', 500000),
    workers:    intVal('rave-workers', 4),
    sample_rate: intVal('rave-sr', 44100),
  }}).catch(() => {});
  try {
    await postJSON('/api/rave/train', {
      name:        document.getElementById('rave-name')?.value  || 'lambert',
      config:      document.getElementById('rave-config')?.value || 'v2',
      n_signal:    intVal('rave-n-signal',  131072),
      batch_size:  intVal('rave-batch',     8),
      n_steps:     intVal('rave-steps',     500000),
      workers:     intVal('rave-workers',   4),
    });
    startRavePoll();
  } catch (e) {
    btn.disabled = false;
    stop.classList.add('hidden');
    showRaveProgress('idle');
  }
});

document.getElementById('rave-stop-btn')?.addEventListener('click', async () => {

  await postJSON('/api/rave/stop', {});
});

function startRavePoll() {
  if (_ravePollIv) return;
  _ravePollIv = setInterval(pollRave, 2000);
}

async function pollRave() {
  try {
    const s = await fetchJSON('/api/rave/status');
    applyRaveStatus(s);
    if (s.status === 'done' || s.status === 'error' || s.status === 'idle') {
      clearInterval(_ravePollIv);
      _ravePollIv = null;
      if (s.status === 'done') {
        loadModelsTab();
        updateStatusBanner({ vae_ready: true });
      }
    }
  } catch (_) {}
}

function applyRaveStatus(s) {
  // Install step
  const installStatus = document.getElementById('rave-install-status');
  const installBtn    = document.getElementById('rave-install-btn');
  if (installStatus) {
    if (s.installed) {
      installStatus.innerHTML = '<span style="color:var(--ok)">✓ acids-rave installed</span>';
      if (installBtn) installBtn.classList.add('hidden');
    } else {
      const err = s.install_error ? ` — ${s.install_error}` : '';
      installStatus.innerHTML = `<span style="color:var(--warn)">Not installed${err}</span>`;
      if (installBtn) installBtn.classList.remove('hidden');
    }
  }

  // Preprocess step
  const prepStatus = document.getElementById('rave-preprocess-status');
  const prepBtn    = document.getElementById('rave-preprocess-btn');
  if (prepStatus) {
    if (s.preprocess_ready) {
      const info = s.preprocess_info || {};
      const hours = info.n_hours ? ` — ${info.n_hours}h of audio` : '';
      prepStatus.innerHTML = `<span style="color:var(--ok)">✓ Preprocessed data ready${hours}</span>`;
      if (prepBtn) { prepBtn.textContent = 'Re-run preprocess'; prepBtn.disabled = false; }
    } else if (s.status === 'preprocessing') {
      prepStatus.innerHTML = '<span style="color:var(--warn)">Preprocessing in progress…</span>';
      if (prepBtn) { prepBtn.textContent = 'Preprocessing…'; prepBtn.disabled = true; }
    } else {
      prepStatus.textContent = 'Not done. Run this before training.';
      if (prepBtn) { prepBtn.textContent = 'Run RAVE preprocess'; prepBtn.disabled = !s.installed; }
    }
  }

  // Train buttons
  const trainBtn = document.getElementById('rave-train-btn');
  const stopBtn  = document.getElementById('rave-stop-btn');
  if (trainBtn) {
    trainBtn.disabled = !s.installed || s.status === 'training';
  }

  // Shared live status bar
  showRaveProgress(s.status, s.step, s.total_steps);

  // Feed last 100 log lines into the shared Train tab log box
  if (s.log_tail?.length) {
    const inner = document.getElementById('train-log-inner');
    if (inner) {
      inner.textContent = s.log_tail.slice(-100).join('\n');
      inner.parentElement.scrollTop = inner.parentElement.scrollHeight;
    }
  }

  // VRAM estimate
  updateRaveVRAMEstimate(s);

  if (s.status === 'done') {
    stopBtn?.classList.add('hidden');
    if (trainBtn) trainBtn.disabled = false;
  } else if (s.status === 'error') {
    stopBtn?.classList.add('hidden');
    if (trainBtn) trainBtn.disabled = false;
  }
}

function showRaveProgress(status, step, total) {
  const statusEl = document.getElementById('live-status');
  const progWrap = document.getElementById('train-progress-wrap');
  const progBar  = document.getElementById('train-progress-bar');
  const progVal  = document.getElementById('train-progress-val');
  const progLabel = document.getElementById('progress-label');

  if (!statusEl) return;

  const colors = { done: 'var(--ok)', error: 'var(--danger)', idle: 'var(--text-dim)' };
  statusEl.style.color = colors[status] || 'var(--warn)';

  if (status === 'training' && total > 0) {
    statusEl.textContent = `RAVE training — step ${(step||0).toLocaleString()} / ${total.toLocaleString()}`;
    if (progWrap) progWrap.style.display = '';
    if (progLabel) progLabel.textContent = 'Steps';
    if (progBar)   progBar.style.width = Math.min(100, ((step||0) / total * 100)).toFixed(1) + '%';
    if (progVal)   progVal.textContent = `${(step||0).toLocaleString()} / ${total.toLocaleString()}`;
  } else if (status === 'preprocessing') {
    statusEl.textContent = 'RAVE preprocessing…';
    if (progWrap) progWrap.style.display = 'none';
  } else if (status === 'installing') {
    statusEl.textContent = 'Installing acids-rave…';
    if (progWrap) progWrap.style.display = 'none';
  } else if (status === 'done') {
    statusEl.textContent = 'RAVE done — model exported and registered';
    if (progWrap) progWrap.style.display = 'none';
  } else if (status === 'error') {
    statusEl.textContent = 'RAVE error — see log';
    if (progWrap) progWrap.style.display = 'none';
  } else {
    statusEl.textContent = 'Idle';
    if (progWrap) progWrap.style.display = 'none';
  }
}

// (appendRaveLog removed — all log writes go through server _rave_log)

async function refreshRaveUI() {
  try {
    const s = await fetchJSON('/api/rave/status');
    applyRaveStatus(s);
    if (s.status === 'training' || s.status === 'preprocessing' || s.status === 'installing') {
      startRavePoll();
    }
  } catch (_) {}
}

function updateRaveVRAMEstimate(s) {
  const est  = s?.vram_estimate;
  const free = window._hwLastGpuFree || 8192;
  const bar  = document.getElementById('rave-vram-bar');
  const val  = document.getElementById('rave-vram-val');
  const warn = document.getElementById('rave-vram-warnings');
  if (!est || !bar) return;
  const mb  = est.total_mb || 0;
  const pct = Math.min(100, mb / free * 100);
  bar.style.width    = pct.toFixed(1) + '%';
  bar.className      = 'bar-fill ' + (pct > 90 ? 'bar-danger' : pct > 75 ? 'bar-warn' : '');
  if (val) val.textContent = `${mb} MB (model ${est.model_mb}, activations ${est.act_mb})`;
  if (warn) {
    warn.innerHTML = pct > 90
      ? '<div class="warn-item err">⚠ May exceed VRAM — reduce batch size</div>'
      : '';
  }
}


// ---------------------------------------------------------------------------
// History tab
// ---------------------------------------------------------------------------

async function loadHistoryTab() {
  await Promise.all([loadHistoryRuns(), loadHistoryGens(), loadHistoryEvents()]);
}

async function loadHistoryRuns() {
  const container = document.getElementById('history-runs-list');
  try {
    const runs = await fetchJSON('/api/history/runs');
    if (!runs.length) {
      container.innerHTML = '<p class="hint">No training runs recorded yet.</p>';
      return;
    }
    const table = document.createElement('table');
    table.className = 'history-table';
    table.innerHTML = `<thead><tr>
      <th>Date</th><th>Stage</th><th>Epochs (done)</th>
      <th>Best VAE val</th><th>Best LSTM val</th>
      <th>Duration</th><th>Device</th><th>Status</th><th>Loss curve</th>
    </tr></thead>`;
    const tbody = document.createElement('tbody');
    runs.forEach(r => {
      const tr = document.createElement('tr');
      tr.className = 'expandable';
      const dur   = r.duration_s != null ? fmtDuration(r.duration_s) : '—';
      const stCls = `run-status-${r.status}`;
      tr.innerHTML = `
        <td>${r.ts_start_str || '?'}</td>
        <td>${r.params?.stage || '?'}</td>
        <td>${r.n_epochs}</td>
        <td>${r.best_vae_val  != null ? r.best_vae_val.toFixed(5)  : '—'}</td>
        <td>${r.best_lstm_val != null ? r.best_lstm_val.toFixed(5) : '—'}</td>
        <td>${dur}</td>
        <td>${r.params?.device || '?'}</td>
        <td class="${stCls}">${r.status}</td>
        <td><canvas class="sparkline-canvas" id="spark-${r.run_id}"></canvas></td>
      `;
      // Expand/collapse row
      const detailRow = document.createElement('tr');
      detailRow.style.display = 'none';
      const detailTd = document.createElement('td');
      detailTd.colSpan = 9;
      detailTd.id = `detail-${r.run_id}`;
      detailRow.appendChild(detailTd);

      tr.addEventListener('click', () => toggleRunDetail(r.run_id, detailRow, detailTd));
      tbody.appendChild(tr);
      tbody.appendChild(detailRow);
    });
    table.appendChild(tbody);
    container.innerHTML = '';
    container.appendChild(table);

    // Draw sparklines after DOM is ready
    requestAnimationFrame(() => {
      runs.forEach(r => drawSparkline(`spark-${r.run_id}`, r));
    });
  } catch (e) {
    container.textContent = 'Failed to load runs: ' + e.message;
  }
}

function drawSparkline(canvasId, run) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  canvas.width  = 120;
  canvas.height = 32;
  const ctx = canvas.getContext('2d');
  // We only have summary data here; full data loaded on expand
  // Show best val losses as coloured dots if available
  ctx.fillStyle = '#1e1e28';
  ctx.fillRect(0, 0, 120, 32);
  if (run.best_vae_val != null) {
    ctx.fillStyle = '#7b6ee8';
    ctx.font = '9px monospace';
    ctx.fillText(`VAE: ${run.best_vae_val.toFixed(4)}`, 4, 12);
  }
  if (run.best_lstm_val != null) {
    ctx.fillStyle = '#4caf7d';
    ctx.fillText(`LSTM: ${run.best_lstm_val.toFixed(5)}`, 4, 26);
  }
}

async function toggleRunDetail(runId, row, td) {
  if (row.style.display !== 'none') {
    row.style.display = 'none';
    return;
  }
  row.style.display = '';
  if (td.dataset.loaded) return;
  td.dataset.loaded = '1';
  td.innerHTML = '<div class="run-detail"><em style="color:var(--text-muted)">Loading…</em></div>';

  try {
    const d = await fetchJSON(`/api/history/run/${runId}`);
    const div = document.createElement('div');
    div.className = 'run-detail';

    // Params grid
    const grid = document.createElement('div');
    grid.className = 'run-detail-grid';
    const params = { ...d.params };
    const hw     = d.hardware?.gpus?.[0] || {};
    const extras = {
      'GPU':       hw.name || '—',
      'VRAM free': hw.vram_free_mb ? hw.vram_free_mb + ' MB' : '—',
      'RAM total': d.hardware?.cpu?.ram_total_gb ? d.hardware.cpu.ram_total_gb + ' GB' : '—',
      'CPU':       d.hardware?.cpu?.model || '—',
      'Run ID':    d.run_id,
      'Started':   d.ts_start_str,
      'Finished':  d.ts_end_str || '—',
      'Duration':  d.duration_s ? fmtDuration(d.duration_s) : '—',
    };
    for (const [k, v] of Object.entries({ ...params, ...extras })) {
      const kv = document.createElement('div');
      kv.className = 'run-detail-kv';
      kv.innerHTML = `<span class="run-detail-key">${k}</span><span class="run-detail-val">${v}</span>`;
      grid.appendChild(kv);
    }
    div.appendChild(grid);

    // Loss chart
    if (d.epochs?.length) {
      const chartWrap = document.createElement('div');
      chartWrap.style.cssText = 'background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:12px;';
      const cvs = document.createElement('canvas');
      cvs.id = `detail-chart-${runId}`;
      cvs.height = 120;
      chartWrap.appendChild(cvs);
      div.appendChild(chartWrap);

      requestAnimationFrame(() => {
        const vaeEpochs  = d.epochs.filter(e => e.stage === 'vae');
        const lstmEpochs = d.epochs.filter(e => e.stage === 'lstm');
        const maxLen     = Math.max(vaeEpochs.length, lstmEpochs.length);
        new Chart(cvs.getContext('2d'), {
          type: 'line',
          data: {
            labels: Array.from({ length: maxLen }, (_, i) => i + 1),
            datasets: [
              { label: 'VAE train',  data: vaeEpochs.map(e => e.train_loss),  borderColor: '#7b6ee8', tension: .3, pointRadius: 0, borderWidth: 1.5, yAxisID: 'yV' },
              { label: 'VAE val',    data: vaeEpochs.map(e => e.val_loss),    borderColor: '#9b8ef8', tension: .3, pointRadius: 0, borderWidth: 1, borderDash: [4,3], yAxisID: 'yV' },
              { label: 'LSTM train', data: lstmEpochs.map(e => e.train_loss), borderColor: '#4caf7d', tension: .3, pointRadius: 0, borderWidth: 1.5, yAxisID: 'yL' },
              { label: 'LSTM val',   data: lstmEpochs.map(e => e.val_loss),   borderColor: '#7de0ae', tension: .3, pointRadius: 0, borderWidth: 1, borderDash: [4,3], yAxisID: 'yL' },
            ],
          },
          options: {
            animation: false, responsive: true, maintainAspectRatio: false,
            plugins: { legend: { labels: { color: '#7a79a0', font: { size: 10 } } } },
            scales: {
              x:  { ticks: { color: '#4a4a68', maxTicksLimit: 10 }, grid: { color: '#1e1e28' } },
              yV: { position: 'left',  ticks: { color: '#7b6ee8', font: { size: 9 } }, grid: { color: '#1e1e28' }, title: { display: true, text: 'VAE', color: '#7b6ee8', font: { size: 9 } } },
              yL: { position: 'right', ticks: { color: '#4caf7d', font: { size: 9 } }, grid: { drawOnChartArea: false }, title: { display: true, text: 'LSTM', color: '#4caf7d', font: { size: 9 } } },
            },
          },
        });
      });
    }

    // VRAM log
    if (d.vram_log?.length) {
      const vwrap = document.createElement('div');
      vwrap.style.cssText = 'background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:12px;';
      const vcvs = document.createElement('canvas');
      vcvs.height = 60;
      vwrap.innerHTML = '<div style="font-size:.72rem;color:var(--text-muted);margin-bottom:4px;text-transform:uppercase;letter-spacing:.08em">VRAM over time (PyTorch alloc)</div>';
      vwrap.appendChild(vcvs);
      div.appendChild(vwrap);
      requestAnimationFrame(() => {
        new Chart(vcvs.getContext('2d'), {
          type: 'line',
          data: {
            labels: d.vram_log.map((_, i) => i),
            datasets: [{ data: d.vram_log.map(v => v.vram_mb), borderColor: '#e8933a', tension: .3, pointRadius: 0, borderWidth: 1.5, fill: true, backgroundColor: 'rgba(232,147,58,.08)' }],
          },
          options: {
            animation: false, responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
              x: { display: false },
              y: { ticks: { color: '#4a4a68', font: { size: 9 } }, grid: { color: '#1e1e28' },
                   title: { display: true, text: 'MB', color: '#4a4a68', font: { size: 9 } } },
            },
          },
        });
      });
    }

    // Error
    if (d.error) {
      const errDiv = document.createElement('div');
      errDiv.className = 'run-error-box';
      errDiv.textContent = d.error;
      div.appendChild(errDiv);
    }

    // Log tail
    if (d.log_lines?.length) {
      const logDiv = document.createElement('div');
      logDiv.style.cssText = 'margin-top:4px';
      logDiv.innerHTML = '<div style="font-size:.72rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px">Training log (last 50 lines)</div>';
      const pre = document.createElement('div');
      pre.className = 'log-box';
      pre.style.height = '140px';
      const inner = document.createElement('div');
      inner.className = 'log-inner';
      inner.textContent = d.log_lines.slice(-50).join('\n');
      pre.appendChild(inner);
      logDiv.appendChild(pre);
      div.appendChild(logDiv);
    }

    td.innerHTML = '';
    td.appendChild(div);
  } catch (e) {
    td.innerHTML = `<div class="run-error-box">Failed to load: ${e.message}</div>`;
  }
}

async function loadHistoryGens() {
  const container = document.getElementById('history-gens-list');
  try {
    const gens = await fetchJSON('/api/history/generations');
    if (!gens.length) {
      container.innerHTML = '<p class="hint">No generations recorded yet.</p>';
      return;
    }
    const table = document.createElement('table');
    table.className = 'history-table';
    table.innerHTML = `<thead><tr>
      <th>Date</th><th>Duration</th><th>Temp</th><th>GL iters</th>
      <th>Seed</th><th>Gen time</th><th>VAE epoch</th><th>Status</th><th>File</th>
    </tr></thead>`;
    const tbody = document.createElement('tbody');
    gens.forEach(g => {
      const p   = g.params || {};
      const tr  = document.createElement('tr');
      const fn  = g.filename || '—';
      const url = fn !== '—' ? `/api/outputs/${fn}` : null;
      tr.innerHTML = `
        <td>${g.ts_str || '?'}</td>
        <td>${p.duration != null ? p.duration + ' s' : '—'}</td>
        <td>${p.temperature != null ? p.temperature.toFixed(2) : '—'}</td>
        <td>${p.griffin_lim_iters || '—'}</td>
        <td>${p.seed != null ? p.seed : 'random'}</td>
        <td>${g.gen_time_s != null ? g.gen_time_s + ' s' : '—'}</td>
        <td>${g.model?.vae_epoch != null ? `ep ${g.model.vae_epoch} (val=${g.model.vae_val_loss?.toFixed(4)})` : '—'}</td>
        <td class="run-status-${g.status}">${g.status}</td>
        <td>${url ? `<audio controls src="${url}" style="width:160px;height:24px"></audio>` : fn}</td>
      `;
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    container.innerHTML = '';
    container.appendChild(table);
  } catch (e) {
    container.textContent = 'Failed to load generations: ' + e.message;
  }
}

async function loadHistoryEvents() {
  const el = document.getElementById('history-events-log');
  try {
    const events = await fetchJSON('/api/history/events?n=80');
    el.textContent = events
      .map(e => `[${e.ts_str || '?'}] ${e.kind}  ${JSON.stringify(
        Object.fromEntries(Object.entries(e).filter(([k]) => !['ts','ts_str','kind','params','result','epochs'].includes(k)))
      )}`)
      .join('\n');
    el.parentElement.scrollTop = el.parentElement.scrollHeight;
  } catch (e) {
    el.textContent = 'Failed: ' + e.message;
  }
}

function fmtDuration(s) {
  if (s < 60)   return `${s}s`;
  if (s < 3600) return `${Math.floor(s/60)}m ${s%60}s`;
  return `${Math.floor(s/3600)}h ${Math.floor((s%3600)/60)}m`;
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
(async () => {
  // Restore persisted config
  try {
    const cfg = await fetchJSON('/api/config');
    if (cfg.rave) {
      if (cfg.rave.name)        setInputVal('rave-name',    cfg.rave.name);
      if (cfg.rave.config)      setInputVal('rave-config',  cfg.rave.config);
      if (cfg.rave.batch_size)  setInputVal('rave-batch',   cfg.rave.batch_size);
      if (cfg.rave.n_steps)     setInputVal('rave-steps',   cfg.rave.n_steps);
      if (cfg.rave.workers)     setInputVal('rave-workers', cfg.rave.workers);
      if (cfg.rave.sample_rate) setInputVal('rave-sr',      cfg.rave.sample_rate);
    }
  } catch (_) {}

  // Load data tab content immediately (it's the default active tab)
  loadDataTab();   // fire-and-forget — don't await, let it load in parallel

  await applySmartDefaults();
  await updateVRAMEstimate();

  try {
    const s = await fetchJSON('/api/train_status');
    await updateStatusBanner(s);
    if (s.status === 'training' || s.status === 'preprocessing') {
      initLossChart();
      startTrainPoll();
      document.getElementById('train-btn').disabled = true;
      document.getElementById('stop-btn').classList.remove('hidden');
    }
  } catch (_) {}

  // Init RAVE UI on the active backend
  await refreshRaveUI();
})();

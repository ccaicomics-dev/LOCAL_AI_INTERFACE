/**
 * LocalAI Platform — Model Manager Frontend
 * Vanilla JS single-page app that talks to /api/localai/* endpoints.
 */

const API = '/api/localai';

// ─── State ───────────────────────────────────────────────────────
const state = {
  hardware: null,
  models: [],
  serverStatus: { running: false },
  loading: false,
  loadingModelPath: null,
  settingsOpen: false,
  settingsModel: null,   // model being configured
  settingsValues: {},    // current form values
  commandPreview: '',
  statusPollTimer: null,
  loadEventSource: null,
  loadStartTime: null,
  elapsedTimer: null,
};

// ─── DOM helpers ─────────────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => [...document.querySelectorAll(sel)];
const el = (tag, attrs = {}, ...children) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') e.className = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v);
  }
  children.forEach(c => e.append(typeof c === 'string' ? document.createTextNode(c) : c));
  return e;
};

// ─── API calls ───────────────────────────────────────────────────
async function fetchJSON(path, opts = {}) {
  const res = await fetch(API + path, opts);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

async function loadHardware() {
  try {
    state.hardware = await fetchJSON('/hardware');
    renderHardwareBar();
  } catch (e) {
    console.error('Hardware detection failed:', e);
  }
}

async function loadModels() {
  try {
    state.models = await fetchJSON('/models');
    renderModelList();
  } catch (e) {
    console.error('Model scan failed:', e);
    showEmptyState(`Failed to scan models: ${e.message}`);
  }
}

async function loadStatus() {
  try {
    const prev = state.serverStatus.running;
    state.serverStatus = await fetchJSON('/status');
    if (prev !== state.serverStatus.running) {
      renderModelList(); // re-render cards when running state changes
    } else {
      updateRunningCard(); // just update live stats
    }
    renderStatusBar();
  } catch (e) {
    // Silent fail — status polling
  }
}

// ─── Hardware bar rendering ──────────────────────────────────────
function renderHardwareBar() {
  const hw = state.hardware;
  if (!hw) return;

  const bar = $('#hardware-bar');
  bar.innerHTML = '';

  if (hw.gpu_name) {
    const vramPct = hw.gpu_vram_mb > 0
      ? Math.round(((hw.gpu_vram_mb - hw.gpu_vram_free_mb) / hw.gpu_vram_mb) * 100)
      : 0;
    const vramUsed = ((hw.gpu_vram_mb - hw.gpu_vram_free_mb) / 1024).toFixed(1);
    const vramTotal = (hw.gpu_vram_mb / 1024).toFixed(0);
    bar.append(hwItem('GPU', hw.gpu_name, vramPct, `${vramUsed} / ${vramTotal} GB`, 'vram'));
  }

  if (hw.ram_mb) {
    const ramPct = Math.round(((hw.ram_mb - hw.ram_free_mb) / hw.ram_mb) * 100);
    const ramUsed = ((hw.ram_mb - hw.ram_free_mb) / 1024).toFixed(0);
    const ramTotal = (hw.ram_mb / 1024).toFixed(0);
    bar.append(hwItem('RAM', `${ramTotal} GB System`, ramPct, `${ramUsed} / ${ramTotal} GB`, 'ram'));
  }

  if (hw.cpu_name) {
    const cpuEl = el('div', { class: 'hw-item' },
      el('div', { class: 'hw-label' }, 'CPU'),
      el('div', { class: 'hw-name' }, hw.cpu_name),
      el('div', { class: 'hw-numbers' }, `${hw.cpu_cores} physical cores`)
    );
    bar.append(cpuEl);
  }
}

function hwItem(label, name, pct, numbers, type) {
  const fill = el('div', { class: `progress-fill progress-fill--${type}`, style: `width:${pct}%` });
  const track = el('div', { class: 'progress-track' }, fill);
  const nums = el('div', { class: 'hw-numbers' }, numbers);
  const barRow = el('div', { class: 'hw-bar-row' }, track, nums);
  return el('div', { class: 'hw-item' },
    el('div', { class: 'hw-label' }, label),
    el('div', { class: 'hw-name' }, name),
    barRow
  );
}

// ─── Model list rendering ────────────────────────────────────────
function renderModelList() {
  const list = $('#model-list');
  list.innerHTML = '';

  if (!state.models.length) {
    showEmptyState('No models found. Add a model directory in Settings, then click Scan.');
    return;
  }

  for (const model of state.models) {
    list.append(renderModelCard(model));
  }
}

function renderModelCard(model) {
  const isLoaded = state.serverStatus.running &&
    state.serverStatus.model_path === model.path;
  const isLoading = state.loading && state.loadingModelPath === model.path;

  const cardClass = isLoaded ? 'model-card model-card--loaded'
    : isLoading ? 'model-card model-card--loading'
    : 'model-card';

  const dotClass = isLoaded ? 'model-status-dot model-status-dot--loaded'
    : isLoading ? 'model-status-dot model-status-dot--loading'
    : 'model-status-dot model-status-dot--idle';

  // Badges
  const badges = [];
  badges.push(el('span', { class: model.is_moe ? 'badge badge--moe' : 'badge badge--dense' },
    model.is_moe ? `MoE ×${model.expert_count || '?'}` : 'Dense'));
  if (model.quantization && model.quantization !== 'unknown') {
    badges.push(el('span', { class: 'badge badge--quant' }, model.quantization));
  }
  badges.push(el('span', { class: 'badge badge--size' }, `${model.size_gb} GB`));
  if (model.fits_in_vram) {
    badges.push(el('span', { class: 'badge badge--fit-ok' }, '✓ VRAM'));
  } else if (model.fits_with_ram) {
    badges.push(el('span', { class: 'badge badge--fit-ram' }, '≈ VRAM+RAM'));
  } else if (state.hardware && state.hardware.gpu_vram_mb > 0) {
    badges.push(el('span', { class: 'badge badge--fit-no' }, '⚠ Too large'));
  }

  const metaEl = el('div', { class: 'model-meta' }, ...badges);

  // Stats for loaded model
  let statsEl = null;
  if (isLoaded && state.serverStatus.running) {
    const st = state.serverStatus;
    const ctxPct = st.context_max > 0
      ? Math.round((st.context_used / st.context_max) * 100) : 0;
    statsEl = el('div', { class: 'model-stats', id: 'running-stats' },
      statRow('Speed', `${st.tokens_per_second} t/s`, st.tokens_per_second / 30, 'speed'),
      statRow('Context', `${st.context_used?.toLocaleString() || 0} / ${st.context_max?.toLocaleString() || 0}`, ctxPct / 100, 'ctx'),
    );
  }

  // Action buttons
  const actions = [];
  if (isLoaded) {
    actions.push(el('button', { class: 'btn btn--danger btn--sm', onclick: handleEject }, '⏏ Eject'));
    actions.push(el('button', {
      class: 'btn btn--ghost btn--sm',
      onclick: () => handleAutoOptimize(model),
      title: 'Run auto-optimization (benchmarks different settings to find fastest config)',
    }, '⚡ Optimize'));
    actions.push(el('button', { class: 'btn btn--ghost btn--sm', onclick: () => openSettings(model) }, '⚙ Settings'));
  } else if (isLoading) {
    actions.push(el('button', { class: 'btn btn--ghost btn--sm', onclick: handleEject }, '✕ Cancel'));
  } else {
    actions.push(el('button', {
      class: 'btn btn--primary btn--sm',
      onclick: () => handleLoad(model),
      disabled: state.loading ? 'true' : null,
    }, '▶ Load'));
    actions.push(el('button', { class: 'btn btn--ghost btn--sm', onclick: () => openSettings(model) }, '⚙'));
  }

  const actionsEl = el('div', { class: 'model-actions' }, ...actions);

  const infoEl = el('div', { class: 'model-info' },
    el('div', { class: 'model-name' }, model.name),
    metaEl,
    ...(statsEl ? [statsEl] : [])
  );

  const bodyEl = el('div', { class: 'model-card-body' },
    el('div', { class: dotClass }),
    infoEl,
    actionsEl
  );

  const card = el('div', { class: cardClass, 'data-path': model.path });
  card.append(bodyEl);
  return card;
}

function statRow(label, value, ratio, type) {
  const fill = el('div', {
    class: `stat-bar-fill stat-bar-fill--${type}`,
    style: `width:${Math.min(ratio * 100, 100)}%`
  });
  return el('div', { class: 'stat-row' },
    el('span', { class: 'stat-label text-muted' }, label),
    el('div', { class: 'stat-bar' }, fill),
    el('span', { class: 'stat-value' }, value)
  );
}

function updateRunningCard() {
  // Live update stats without full re-render
  const statsEl = $('#running-stats');
  if (!statsEl || !state.serverStatus.running) return;
  const st = state.serverStatus;
  const ctxPct = st.context_max > 0 ? Math.round((st.context_used / st.context_max) * 100) : 0;
  statsEl.innerHTML = '';
  statsEl.append(
    statRow('Speed', `${st.tokens_per_second} t/s`, st.tokens_per_second / 30, 'speed'),
    statRow('Context', `${st.context_used?.toLocaleString() || 0} / ${st.context_max?.toLocaleString() || 0}`, ctxPct / 100, 'ctx'),
  );
}

function showEmptyState(msg) {
  const list = $('#model-list');
  list.innerHTML = '';
  list.append(el('div', { class: 'empty-state' },
    el('h3', {}, 'No Models Found'),
    el('p', {}, msg)
  ));
}

// ─── Status bar ──────────────────────────────────────────────────
function renderStatusBar() {
  const bar = $('#status-bar');
  const st = state.serverStatus;
  if (st.running) {
    bar.innerHTML = `
      <span class="text-green">● Server Running</span>
      <span>|</span>
      <span>${st.model_name}</span>
      <span>|</span>
      <span>llama-server :8001</span>
      <span>|</span>
      <span>API: <a href="http://localhost:3000/api" style="color:var(--blue)" target="_blank">:3000/api</a></span>
    `;
  } else {
    bar.innerHTML = `
      <span class="text-muted">● Server Idle</span>
      <span>|</span>
      <span>LocalAI Platform</span>
      <span>|</span>
      <span>API: <a href="http://localhost:3000/api" style="color:var(--blue)" target="_blank">:3000/api</a></span>
    `;
  }
}

// ─── Load model ──────────────────────────────────────────────────
function handleLoad(model) {
  state.loading = true;
  state.loadingModelPath = model.path;
  state.loadStartTime = Date.now();

  renderModelList();
  showLoadOverlay(model);

  // Connect SSE
  const url = `${API}/load`;
  fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model_path: model.path,
      override_flags: state.settingsValues[model.path] || null,
    }),
  }).then(async res => {
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const event = JSON.parse(line.slice(6));
            handleLoadEvent(event);
          } catch {}
        }
      }
    }
  }).catch(e => {
    handleLoadEvent({ step: 'error', message: e.message });
  });

  startElapsedTimer();
}

const LOAD_STEPS = [
  { key: 'ejecting',   label: 'Stopping previous model' },
  { key: 'inspecting', label: 'Reading model metadata' },
  { key: 'optimizing', label: 'Computing optimal flags' },
  { key: 'launching',  label: 'Launching llama-server' },
  { key: 'loading',    label: 'Loading model weights' },
];

let _loadStepState = {};

function showLoadOverlay(model) {
  _loadStepState = {};
  const overlay = $('#load-overlay');
  overlay.classList.remove('hidden');

  $('#load-model-name').textContent = model.name;
  renderLoadSteps('inspecting');
  $('#load-progress-fill').style.width = '0%';
}

function hideLoadOverlay() {
  $('#load-overlay').classList.add('hidden');
  clearInterval(state.elapsedTimer);
  state.elapsedTimer = null;
}

function handleLoadEvent(event) {
  const step = event.step;
  if (step === 'done') return;

  if (step === 'error') {
    hideLoadOverlay();
    state.loading = false;
    state.loadingModelPath = null;
    renderModelList();
    showError(event.message || 'Load failed');
    return;
  }

  if (step === 'ready') {
    $('#load-progress-fill').style.width = '100%';
    renderLoadSteps('ready');
    setTimeout(() => {
      hideLoadOverlay();
      state.loading = false;
      loadStatus();
      renderModelList();
    }, 1200);
    return;
  }

  // Update progress bar
  if (step === 'loading' && event.progress) {
    $('#load-progress-fill').style.width = `${event.progress}%`;
  }

  renderLoadSteps(step);
}

function renderLoadSteps(currentStep) {
  const container = $('#load-steps');
  container.innerHTML = '';

  let reached = false;
  for (const s of LOAD_STEPS) {
    if (s.key === currentStep) reached = true;
    const isDone = !reached && s.key !== currentStep;
    const isActive = s.key === currentStep;

    const iconText = isDone ? '✓' : isActive ? '⟳' : '○';
    const iconClass = isDone ? 'step-icon--done' : isActive ? 'step-icon--active' : 'step-icon--pending';
    const textClass = isDone ? 'step-text--done' : isActive ? 'step-text--active' : 'step-text--pending';

    container.append(
      el('div', { class: 'load-step' },
        el('div', { class: `step-icon ${iconClass}` }, iconText),
        el('span', { class: textClass }, s.label)
      )
    );
  }
}

function startElapsedTimer() {
  clearInterval(state.elapsedTimer);
  state.elapsedTimer = setInterval(() => {
    const elapsedEl = $('#load-elapsed');
    if (elapsedEl && state.loadStartTime) {
      const sec = Math.round((Date.now() - state.loadStartTime) / 1000);
      elapsedEl.textContent = `${sec}s elapsed`;
    }
  }, 1000);
}

// ─── Eject model ─────────────────────────────────────────────────
async function handleEject() {
  try {
    hideLoadOverlay();
    state.loading = false;
    state.loadingModelPath = null;
    await fetchJSON('/eject', { method: 'POST' });
    await loadStatus();
    renderModelList();
  } catch (e) {
    showError(`Eject failed: ${e.message}`);
  }
}

// ─── Auto-optimize ───────────────────────────────────────────────
function handleAutoOptimize(model) {
  if (!confirm(
    `Run auto-optimization for ${model.name}?\n\n` +
    `This will restart the model multiple times with different settings ` +
    `to find the fastest configuration.\n\n` +
    `The model will be unavailable during optimization (2-5 minutes).\n` +
    `Chat will pause until complete.`
  )) return;

  // Show the optimize overlay
  const overlay = $('#load-overlay');
  overlay.classList.remove('hidden');
  $('#load-model-name').textContent = `Optimizing: ${model.name}`;
  $('#load-progress-fill').style.width = '0%';

  state.loadStartTime = Date.now();
  startElapsedTimer();

  const optimizeLog = [];
  let totalExperiments = 10;
  let currentExperiment = 0;

  // Render optimize-specific steps
  const stepsEl = $('#load-steps');
  stepsEl.innerHTML = '';
  stepsEl.append(
    el('div', { class: 'load-step' },
      el('div', { class: 'step-icon step-icon--active' }, '⟳'),
      el('span', { class: 'step-text--active' }, 'Running baseline benchmark...')
    )
  );

  // Connect SSE
  fetch(`${API}/auto-optimize`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model_path: model.path, max_iterations: 10 }),
  }).then(async res => {
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        try {
          const event = JSON.parse(line.slice(6));
          handleOptimizeEvent(event, stepsEl, optimizeLog);
        } catch {}
      }
    }
  }).catch(e => {
    handleOptimizeEvent({ step: 'error', message: e.message }, stepsEl, optimizeLog);
  });
}

function handleOptimizeEvent(event, stepsEl, logArr) {
  const step = event.step;
  if (step === 'done') return;

  if (step === 'error') {
    hideLoadOverlay();
    showError(event.message || 'Optimization failed');
    loadStatus();
    renderModelList();
    return;
  }

  if (step === 'baseline_done') {
    stepsEl.innerHTML = '';
    stepsEl.append(
      el('div', { class: 'load-step' },
        el('div', { class: 'step-icon step-icon--done' }, '✓'),
        el('span', { class: 'step-text--done' }, `Baseline: ${event.tps} t/s`)
      )
    );
  }

  if (step === 'experiment') {
    $('#load-progress-fill').style.width =
      `${Math.round((event.iteration / 10) * 100)}%`;
    // Update last step to show current experiment
    const existing = stepsEl.querySelectorAll('.opt-experiment');
    if (existing.length > 5) existing[0].remove(); // Keep last 5 visible
    stepsEl.append(
      el('div', { class: 'load-step opt-experiment' },
        el('div', { class: 'step-icon step-icon--active' }, '⟳'),
        el('span', { class: 'step-text--active' }, event.message)
      )
    );
  }

  if (step === 'improvement') {
    logArr.push(event);
    const last = stepsEl.querySelector('.opt-experiment:last-child');
    if (last) {
      last.querySelector('.step-icon').className = 'step-icon step-icon--done';
      last.querySelector('.step-icon').textContent = '✓';
      last.querySelector('span').className = 'step-text--done';
      last.querySelector('span').textContent = event.message;
    }
  }

  if (step === 'reverted') {
    const last = stepsEl.querySelector('.opt-experiment:last-child');
    if (last) {
      last.querySelector('.step-icon').className = 'step-icon step-icon--pending';
      last.querySelector('.step-icon').textContent = '✗';
      last.querySelector('span').className = 'step-text--pending';
      last.querySelector('span').textContent = event.message;
    }
  }

  if (step === 'complete') {
    $('#load-progress-fill').style.width = '100%';
    const report = event.report || {};
    stepsEl.append(
      el('div', { class: 'load-step', style: 'margin-top:12px' },
        el('div', { class: 'step-icon step-icon--done' }, '★'),
        el('span', { class: 'step-text--done', style: 'font-weight:600' },
          `Done! ${report.baseline_tps} → ${report.final_tps} t/s (+${report.total_improvement_pct}%)`
        )
      )
    );
    setTimeout(() => {
      hideLoadOverlay();
      loadStatus();
      renderModelList();
    }, 3000);
  }
}

// ─── Settings panel ──────────────────────────────────────────────
async function openSettings(model) {
  state.settingsModel = model;
  state.settingsOpen = true;

  const panel = $('#settings-panel');
  panel.classList.add('open');
  $('#settings-model-name').textContent = model.name;

  const saved = state.settingsValues[model.path] || {};

  // Populate form fields
  $('#set-ctx').value = saved.ctx_size || '';
  $('#set-threads').value = saved.threads || '';
  $('#set-gpu-layers').value = saved.gpu_layers || '999';
  $('#set-moe').checked = saved.moe_offload !== false;
  $('#set-moe').disabled = !model.is_moe;
  $('#set-flash').checked = saved.flash_attn !== false;
  $('#set-kv').value = saved.kv_quant || 'q8_0';

  await refreshCommandPreview();
}

function closeSettings() {
  state.settingsOpen = false;
  $('#settings-panel').classList.remove('open');
}

async function saveSettings() {
  if (!state.settingsModel) return;
  const model = state.settingsModel;
  const overrides = {
    ctx_size: parseInt($('#set-ctx').value) || null,
    threads: parseInt($('#set-threads').value) || null,
    gpu_layers: parseInt($('#set-gpu-layers').value) || 999,
    moe_offload: $('#set-moe').checked,
    flash_attn: $('#set-flash').checked,
    kv_quant: $('#set-kv').value,
  };
  // Remove null values
  for (const k of Object.keys(overrides)) {
    if (overrides[k] === null) delete overrides[k];
  }
  state.settingsValues[model.path] = overrides;
  closeSettings();
}

async function loadWithSettings() {
  await saveSettings();
  const model = state.settingsModel;
  closeSettings();
  if (model) handleLoad(model);
}

async function refreshCommandPreview() {
  const model = state.settingsModel;
  if (!model) return;

  const overrides = {
    ctx_size: parseInt($('#set-ctx').value) || null,
    threads: parseInt($('#set-threads').value) || null,
    gpu_layers: parseInt($('#set-gpu-layers').value) || 999,
    flash_attn: $('#set-flash').checked,
    kv_quant: $('#set-kv').value,
  };

  try {
    const result = await fetchJSON('/optimize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_path: model.path, overrides }),
    });
    $('#command-preview').textContent = result.command_preview || '';
    if (!$('#set-ctx').value && result.ctx_size) {
      $('#ctx-auto-hint').textContent = `auto: ${result.ctx_size.toLocaleString()} tokens`;
    }
  } catch (e) {
    $('#command-preview').textContent = `Error: ${e.message}`;
  }
}

// ─── App-level settings (model dirs) ────────────────────────────
async function openAppSettings() {
  try {
    const settings = await fetchJSON('/settings');
    const dirs = (settings.model_dirs || []).join('\n');
    const newDirs = prompt('Model directories (one per line):', dirs);
    if (newDirs === null) return;
    const updated = {
      ...settings,
      model_dirs: newDirs.split('\n').map(d => d.trim()).filter(Boolean),
    };
    await fetchJSON('/settings', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(updated),
    });
    await loadModels();
  } catch (e) {
    showError(`Settings error: ${e.message}`);
  }
}

// ─── Error display ───────────────────────────────────────────────
function showError(message) {
  const container = $('#model-list');
  const toast = el('div', {
    class: 'model-card model-card--error',
    style: 'padding:16px; color:var(--red); font-size:13px;'
  }, `⚠ ${message}`);
  container.prepend(toast);
  setTimeout(() => toast.remove(), 8000);
}

// ─── Init ────────────────────────────────────────────────────────
async function init() {
  // Wire up static UI events
  $('#btn-scan').addEventListener('click', loadModels);
  $('#btn-app-settings').addEventListener('click', openAppSettings);
  $('#btn-close-settings').addEventListener('click', closeSettings);
  $('#btn-save-settings').addEventListener('click', saveSettings);
  $('#btn-load-with-settings').addEventListener('click', loadWithSettings);
  $('#btn-reset-settings').addEventListener('click', () => {
    if (state.settingsModel) {
      delete state.settingsValues[state.settingsModel.path];
      openSettings(state.settingsModel);
    }
  });
  $('#btn-cancel-load').addEventListener('click', handleEject);

  // Settings form: refresh preview on change
  ['#set-ctx', '#set-threads', '#set-gpu-layers', '#set-kv'].forEach(sel => {
    $(sel)?.addEventListener('input', refreshCommandPreview);
  });
  ['#set-moe', '#set-flash'].forEach(sel => {
    $(sel)?.addEventListener('change', refreshCommandPreview);
  });

  // Initial data load
  await loadHardware();
  await loadModels();
  await loadStatus();
  renderStatusBar();

  // Poll status every 5 seconds
  state.statusPollTimer = setInterval(loadStatus, 5000);

  // Refresh hardware every 30 seconds (VRAM usage changes)
  setInterval(loadHardware, 30000);
}

document.addEventListener('DOMContentLoaded', init);

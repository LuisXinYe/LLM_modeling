/**
 * rl-perf SPA — Application Logic
 *
 * Handles: accordion, tabs, form state, API calls, Plotly charts.
 */

/* ── State ────────────────────────────────────────────── */
const state = {
  templates: {},
  hardware: {},
  presets: {},
  lastResult: null,
  lastSearch: null,
  hasRun: false,
  modified: { model: false, hardware: false, rl: false, search: false },
};

/* ── DOM refs ─────────────────────────────────────────── */
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

/* ── Init ─────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', async () => {
  initAccordion();
  initTabs();
  initDrawer();
  initSegmentedControls();
  initConditionalFields();
  initAutoComputed();
  initButtons();
  initJsonEditor();
  await loadConfigs();
  initTemplateListener();
});

/* ── Accordion ────────────────────────────────────────── */
function initAccordion() {
  // Open the first section by default
  const first = $('.accordion-section[data-section="model"]');
  if (first) first.setAttribute('data-open', '');

  $$('.accordion-trigger').forEach((trigger) => {
    trigger.addEventListener('click', () => {
      const section = trigger.closest('.accordion-section');
      const isOpen = section.hasAttribute('data-open');

      // Close all
      $$('.accordion-section').forEach((s) => {
        s.removeAttribute('data-open');
        s.querySelector('.accordion-trigger').setAttribute('aria-expanded', 'false');
      });

      // Open clicked (if it was closed)
      if (!isOpen) {
        section.setAttribute('data-open', '');
        trigger.setAttribute('aria-expanded', 'true');
      }
    });
  });
}

/* ── Tabs ─────────────────────────────────────────────── */
function initTabs() {
  $$('.chart-tab').forEach((tab) => {
    tab.addEventListener('click', () => {
      $$('.chart-tab').forEach((t) => t.classList.remove('active'));
      $$('.chart-panel').forEach((p) => p.classList.remove('active'));
      tab.classList.add('active');
      $(`#panel-${tab.dataset.tab}`).classList.add('active');
    });
  });
}

/* ── Drawer (mobile) ──────────────────────────────────── */
function initDrawer() {
  const toggle = $('#drawer-toggle');
  const panel = $('#config-panel');
  const overlay = $('#drawer-overlay');

  if (!toggle) return;

  toggle.addEventListener('click', () => {
    panel.classList.toggle('open');
    overlay.classList.toggle('visible');
  });

  overlay.addEventListener('click', () => {
    panel.classList.remove('open');
    overlay.classList.remove('visible');
  });
}

/* ── Segmented Controls ───────────────────────────────── */
function initSegmentedControls() {
  // Model source
  const modelSource = $('#model-source');
  if (modelSource) {
    modelSource.querySelectorAll('.seg-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        modelSource.querySelectorAll('.seg-btn').forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        const val = btn.dataset.value;

        $('#field-template').classList.toggle('hidden', val !== 'template');
        $('#field-hf').classList.toggle('hidden', val !== 'huggingface');

        markModified('model');
      });
    });
  }

  // Search mode
  const searchMode = $('#search-mode');
  if (searchMode) {
    searchMode.querySelectorAll('.seg-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        searchMode.querySelectorAll('.seg-btn').forEach((b) => b.classList.remove('active'));
        btn.classList.add('active');
        const val = btn.dataset.value;

        $('#pareto-fields').classList.toggle('hidden', val !== 'pareto');
        $('#sensitivity-fields').classList.toggle('hidden', val !== 'sensitivity');

        updateSearchSummary();
        markModified('search');
      });
    });
  }
}

/* ── Conditional Fields ───────────────────────────────── */
function initConditionalFields() {
  const attnSelect = $('#attention-type');
  const ffnSelect = $('#ffn-type');
  const residualSelect = $('#residual-type');
  const specDecode = $('#spec-decode');

  if (attnSelect) {
    attnSelect.addEventListener('change', () => {
      updateConditionalFields();
      markModified('model');
    });
  }
  if (ffnSelect) {
    ffnSelect.addEventListener('change', () => {
      updateConditionalFields();
      markModified('model');
    });
  }
  if (residualSelect) {
    residualSelect.addEventListener('change', () => {
      updateConditionalFields();
      markModified('model');
    });
  }
  if (specDecode) {
    specDecode.addEventListener('change', () => {
      $('#mtp-fields').classList.toggle('hidden', !specDecode.checked);
      markModified('rl');
    });
  }

  // Track all input changes for modification dots
  $$('.accordion-section[data-section="model"] input, .accordion-section[data-section="model"] select').forEach((el) => {
    el.addEventListener('change', () => markModified('model'));
  });
  $$('.accordion-section[data-section="hardware"] input, .accordion-section[data-section="hardware"] select').forEach((el) => {
    el.addEventListener('change', () => markModified('hardware'));
  });
  $$('.accordion-section[data-section="rl"] input, .accordion-section[data-section="rl"] select').forEach((el) => {
    el.addEventListener('change', () => markModified('rl'));
  });
  $$('.accordion-section[data-section="search"] input, .accordion-section[data-section="search"] select').forEach((el) => {
    el.addEventListener('change', () => markModified('search'));
  });
}

function updateConditionalFields() {
  const attn = $('#attention-type').value;
  const ffn = $('#ffn-type').value;
  const residual = $('#residual-type').value;

  $('#mla-fields').classList.toggle('hidden', attn !== 'MLA');
  $('#swa-fields').classList.toggle('hidden', attn !== 'SWA');
  $('#moe-fields').classList.toggle('hidden', ffn !== 'MoE');
  $('#mhc-fields').classList.toggle('hidden', residual !== 'mHC');
}

/* ── Auto-computed Fields ─────────────────────────────── */
function initAutoComputed() {
  // DP = total_devices / (TP * PP * EP)
  const recomputeDP = () => {
    const total = intVal('total-devices');
    const tp = intVal('par-tp');
    const pp = intVal('par-pp');
    const ep = intVal('par-ep');
    const divisor = tp * pp * ep;
    const dp = divisor > 0 ? Math.floor(total / divisor) : total;
    $('#par-dp').value = dp;

    // Nodes
    const perNode = intVal('devices-per-node');
    const nodes = perNode > 0 ? Math.ceil(total / perNode) : 1;
    $('#num-nodes').value = nodes;

    updateHardwareSummary();
  };

  ['total-devices', 'par-tp', 'par-pp', 'par-ep'].forEach((id) => {
    const el = $(`#${id}`);
    if (el) el.addEventListener('input', recomputeDP);
  });

  // Total responses = prompts * group_size
  const recomputeResponses = () => {
    const prompts = intVal('total-prompts');
    const group = intVal('group-size');
    $('#total-responses').value = prompts * group;
    updateRLSummary();
  };

  ['total-prompts', 'group-size'].forEach((id) => {
    const el = $(`#${id}`);
    if (el) el.addEventListener('input', recomputeResponses);
  });

  // Hardware profile change
  $('#hw-profile').addEventListener('change', () => {
    const prof = state.hardware[$('#hw-profile').value];
    if (prof) {
      $('#devices-per-node').value = prof.devices_per_node;
      recomputeDP();
    }
    markModified('hardware');
  });

  // Deploy mode change
  $('#deploy-mode').addEventListener('change', () => updateRLSummary());
  $('#ref-model').addEventListener('change', () => updateRLSummary());
}

/* ── Summary Updaters ─────────────────────────────────── */
function updateModelSummary() {
  const name = $('#model-name').value || 'Custom';
  const layers = $('#num-layers').value;
  const attn = $('#attention-type').value;
  const dtype = $('#dtype').value;
  $('#model-summary').textContent = `${name} \u00b7 ${layers}L \u00b7 ${attn} \u00b7 ${dtype}`;
}

function updateHardwareSummary() {
  const hw = $('#hw-profile').value;
  const devices = $('#total-devices').value;
  const nodes = $('#num-nodes').value;
  $('#hardware-summary').textContent = `${hw} \u00b7 ${devices} devices \u00b7 ${nodes} node${nodes > 1 ? 's' : ''}`;
}

function updateRLSummary() {
  const prompts = intVal('total-prompts');
  const promptStr = prompts >= 1000 ? `${Math.round(prompts / 1000)}k` : prompts;
  const group = $('#group-size').value;
  const mode = $('#deploy-mode').value === 'colocated' ? 'colocated' : 'separate';
  const ref = $('#ref-model').checked ? 'ref model' : 'no ref';
  $('#rl-summary').textContent = `${promptStr} prompts \u00b7 grp=${group} \u00b7 ${mode} \u00b7 ${ref}`;
}

function updateSearchSummary() {
  const mode = getSearchMode();
  if (mode === 'pareto') {
    const counts = $('#device-counts').value.split(',').filter(Boolean).length;
    $('#search-summary').textContent = `Pareto \u00b7 ${counts} device counts`;
  } else {
    const param = $('#sweep-param').value;
    $('#search-summary').textContent = `Sensitivity \u00b7 ${param}`;
  }
}

function markModified(section) {
  state.modified[section] = true;
  const dot = $(`.accordion-section[data-section="${section}"] .accordion-dot`);
  if (dot) {
    dot.classList.remove('dot-default');
    dot.classList.add('dot-modified');
  }

  // Update summaries
  if (section === 'model') updateModelSummary();
  if (section === 'hardware') updateHardwareSummary();
  if (section === 'rl') updateRLSummary();
  if (section === 'search') updateSearchSummary();
}

function resetModified() {
  Object.keys(state.modified).forEach((k) => {
    state.modified[k] = false;
    const dot = $(`.accordion-section[data-section="${k}"] .accordion-dot`);
    if (dot) {
      dot.classList.remove('dot-modified');
      dot.classList.add('dot-default');
    }
  });
}

/* ── Config Loading ───────────────────────────────────── */
async function loadConfigs() {
  try {
    const [modelsResp, hwResp, presetsResp] = await Promise.all([
      fetch('/api/models'),
      fetch('/api/hardware'),
      fetch('/api/presets'),
    ]);
    const modelsData = await modelsResp.json();
    const hwData = await hwResp.json();
    const presetsData = await presetsResp.json();

    state.templates = modelsData.templates || {};
    state.hardware = hwData.profiles || {};
    state.presets = presetsData.presets || {};

    // Apply first hardware
    const firstHW = Object.keys(state.hardware)[0];
    if (firstHW && state.hardware[firstHW]) {
      $('#devices-per-node').value = state.hardware[firstHW].devices_per_node;
    }

    // If presets available, apply the first preset and auto-run
    const firstPreset = Object.keys(state.presets)[0];
    if (firstPreset) {
      const templateVal = $('#model-template').value;
      if (state.presets[templateVal]) {
        applyPreset(state.presets[templateVal]);
      } else {
        applyPreset(state.presets[firstPreset]);
      }
      // Auto-run prediction on load
      await runAnalysis();
    } else {
      // Fallback: apply first model template
      const firstTemplate = Object.keys(state.templates)[0];
      if (firstTemplate) {
        applyModelTemplate(firstTemplate);
      }
    }
  } catch (e) {
    console.error('Failed to load configs:', e);
  }
}

function applyModelTemplate(name) {
  const t = state.templates[name];
  if (!t) return;

  $('#model-name').value = t.name;
  $('#hidden-size').value = t.hidden_size;
  $('#vocab-size').value = t.vocab_size;
  $('#num-layers').value = t.num_layers;
  $('#dtype').value = t.dtype;

  if (t.layer) {
    $('#attention-type').value = t.layer.attention || 'GQA';
    $('#ffn-type').value = t.layer.ffn || 'SwiGLU';
    $('#residual-type').value = t.layer.residual || 'standard';
    $('#num-heads').value = t.layer.num_heads || 32;
    $('#num-kv-heads').value = t.layer.num_kv_heads || 8;
    $('#head-dim').value = t.layer.head_dim || 128;
    $('#intermediate-size').value = t.layer.intermediate_size || 14336;
    $('#num-experts').value = t.layer.num_experts || 1;
    $('#top-k').value = t.layer.top_k || 1;
    $('#shared-experts').value = t.layer.num_shared_experts || 0;
    $('#expert-intermediate-size').value = t.layer.expert_intermediate_size || 0;
    $('#shared-intermediate-size').value = t.layer.shared_intermediate_size || 0;
    $('#kv-compression-dim').value = t.layer.kv_compression_dim || 0;
    $('#query-compression-dim').value = t.layer.query_compression_dim || 0;
    $('#rope-dim').value = t.layer.rope_dim || 0;
    $('#window-size').value = t.layer.window_size || 0;
    $('#mhc-expansion').value = t.layer.mhc_expansion || 4;
  }

  updateConditionalFields();
  updateModelSummary();
}

function initTemplateListener() {
  $('#model-template').addEventListener('change', async (e) => {
    const name = e.target.value;
    // If a full preset exists for this template, apply it and auto-run
    if (state.presets[name]) {
      applyPreset(state.presets[name]);
      await runAnalysis();
    } else {
      applyModelTemplate(name);
      markModified('model');
    }
  });
}

/* ── Buttons ──────────────────────────────────────────── */
function initButtons() {
  // Run Analysis
  $('#run-btn').addEventListener('click', runAnalysis);

  // Run Search
  $('#run-search-btn').addEventListener('click', runSearch);

  // HF Load
  $('#hf-load-btn').addEventListener('click', loadHFModel);

  // Error close
  $('#error-close').addEventListener('click', () => {
    $('#error-banner').classList.add('hidden');
  });
}

/* ── Build Request Payloads ───────────────────────────── */
function buildPredictRequest() {
  return {
    model: {
      name: $('#model-name').value,
      hidden_size: intVal('hidden-size'),
      vocab_size: intVal('vocab-size'),
      num_layers: intVal('num-layers'),
      dtype: $('#dtype').value,
      layer: {
        attention: $('#attention-type').value,
        num_heads: intVal('num-heads'),
        num_kv_heads: intVal('num-kv-heads'),
        head_dim: intVal('head-dim'),
        ffn: $('#ffn-type').value,
        intermediate_size: intVal('intermediate-size'),
        residual: $('#residual-type').value,
        num_experts: intVal('num-experts'),
        top_k: intVal('top-k'),
        num_shared_experts: intVal('shared-experts'),
        expert_intermediate_size: intVal('expert-intermediate-size'),
        shared_intermediate_size: intVal('shared-intermediate-size'),
        kv_compression_dim: intVal('kv-compression-dim'),
        query_compression_dim: intVal('query-compression-dim'),
        rope_dim: intVal('rope-dim'),
        window_size: intVal('window-size'),
        mhc_expansion: intVal('mhc-expansion'),
      },
    },
    hardware: $('#hw-profile').value,
    total_devices: intVal('total-devices'),
    parallelism: {
      tp: intVal('par-tp'),
      pp: intVal('par-pp'),
      dp: intVal('par-dp'),
      ep: intVal('par-ep'),
      cp: intVal('par-cp'),
      cp_type: $('#cp-type').value,
      sp: $('#par-sp').checked,
      zero_stage: intVal('zero-stage'),
      pp_schedule: $('#pp-schedule').value,
      recompute_attention: $('#recompute-attn').checked,
      full_recomputation: $('#full-recompute').checked,
      optimizer_offload: $('#opt-offload').checked,
      activation_offload: $('#act-offload').checked,
    },
    rl: {
      total_prompts: intVal('total-prompts'),
      group_size: intVal('group-size'),
      avg_prompt_len: intVal('avg-prompt-len'),
      avg_response_len: intVal('avg-response-len'),
      max_response_len: intVal('max-response-len'),
      std_response_len: intValOrNull('std-response-len'),
      train_micro_batch_size: intVal('train-mbs'),
      gradient_accumulation_steps: intVal('grad-accum'),
      gen_batch_size: intVal('gen-batch-size'),
      colocated: $('#deploy-mode').value === 'colocated',
      reference_model: $('#ref-model').checked,
      ref_offload_cpu: $('#ref-offload').checked,
      use_speculative_decoding: $('#spec-decode').checked,
      mtp_acceptance_len: $('#spec-decode').checked ? intValOrNull('mtp-acceptance-len') : null,
    },
  };
}

function getSearchMode() {
  const active = $('#search-mode .seg-btn.active');
  return active ? active.dataset.value : 'pareto';
}

function buildSearchRequest() {
  const base = buildPredictRequest();
  const mode = getSearchMode();

  if (mode === 'pareto') {
    base.search = {
      mode: 'pareto',
      device_counts: $('#device-counts').value.split(',').map((s) => parseInt(s.trim(), 10)).filter((n) => !isNaN(n)),
      optimization_target: $('#opt-target').value,
    };
  } else {
    base.search = {
      mode: 'sensitivity',
      sweep_param: $('#sweep-param').value,
      sweep_values: $('#sweep-values').value.split(',').map((s) => parseInt(s.trim(), 10)).filter((n) => !isNaN(n)),
    };
  }
  return base;
}

/* ── API Calls ────────────────────────────────────────── */
async function runAnalysis() {
  const btn = $('#run-btn');
  const label = btn.querySelector('.btn-label');
  const spinner = btn.querySelector('.btn-spinner');

  setLoading(true);
  btn.disabled = true;
  label.textContent = 'Running...';
  spinner.classList.remove('hidden');
  hideError();

  try {
    const resp = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(buildPredictRequest()),
    });

    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.detail || `Server error ${resp.status}`);
    }

    const data = await resp.json();
    state.lastResult = data;
    state.hasRun = true;
    resetModified();

    renderKPIs(data.kpis, data.memory);
    renderTimeline(data.timeline);
    renderMemory(data.memory);
    renderTopology(data.topology);
    hideEmptyState();
  } catch (e) {
    showError(e.message);
    renderErrorKPIs();
  } finally {
    setLoading(false);
    btn.disabled = false;
    label.textContent = 'Run Analysis';
    spinner.classList.add('hidden');
  }
}

async function runSearch() {
  const btn = $('#run-search-btn');
  btn.disabled = true;
  btn.textContent = 'Searching...';
  hideError();

  try {
    const resp = await fetch('/api/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(buildSearchRequest()),
    });

    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.detail || `Server error ${resp.status}`);
    }

    const data = await resp.json();
    state.lastSearch = data;
    state.hasRun = true;
    hideEmptyState();

    renderSearchResults(data, getSearchMode());

    // Switch to search results tab
    $$('.chart-tab').forEach((t) => t.classList.remove('active'));
    $$('.chart-panel').forEach((p) => p.classList.remove('active'));
    $('.chart-tab[data-tab="search-results"]').classList.add('active');
    $('#panel-search-results').classList.add('active');
  } catch (e) {
    showError(e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Run Search';
  }
}

async function loadHFModel() {
  const btn = $('#hf-load-btn');
  const modelId = $('#hf-model-id').value.trim();
  const status = $('#hf-status');

  if (!modelId) return;

  btn.disabled = true;
  btn.textContent = 'Loading...';
  status.classList.remove('hidden', 'error', 'success');
  status.textContent = 'Fetching config...';

  try {
    const resp = await fetch('/api/hf-import', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_id: modelId }),
    });

    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.detail || `Error ${resp.status}`);
    }

    const data = await resp.json();
    applyHFData(data);

    status.classList.add('success');
    status.textContent = `Loaded ${data.name}`;
    markModified('model');
  } catch (e) {
    status.classList.add('error');
    status.textContent = e.message;
  } finally {
    btn.disabled = false;
    btn.textContent = 'Load';
  }
}

function applyHFData(data) {
  $('#model-name').value = data.name || '';
  $('#hidden-size').value = data.hidden_size || 0;
  $('#vocab-size').value = data.vocab_size || 0;
  $('#num-layers').value = data.num_layers || 0;
  $('#dtype').value = data.dtype || 'bf16';

  if (data.layer) {
    $('#attention-type').value = data.layer.attention || 'GQA';
    $('#ffn-type').value = data.layer.ffn || 'SwiGLU';
    $('#residual-type').value = data.layer.residual || 'standard';
    $('#num-heads').value = data.layer.num_heads || 32;
    $('#num-kv-heads').value = data.layer.num_kv_heads || 8;
    $('#head-dim').value = data.layer.head_dim || 128;
    $('#intermediate-size').value = data.layer.intermediate_size || 14336;
    $('#num-experts').value = data.layer.num_experts || 1;
    $('#top-k').value = data.layer.top_k || 1;
    $('#shared-experts').value = data.layer.num_shared_experts || 0;
    $('#expert-intermediate-size').value = data.layer.expert_intermediate_size || 0;
    $('#shared-intermediate-size').value = data.layer.shared_intermediate_size || 0;
    $('#kv-compression-dim').value = data.layer.kv_compression_dim || 0;
    $('#query-compression-dim').value = data.layer.query_compression_dim || 0;
    $('#rope-dim').value = data.layer.rope_dim || 0;
    $('#window-size').value = data.layer.window_size || 0;
    $('#mhc-expansion').value = data.layer.mhc_expansion || 4;
  }

  updateConditionalFields();
  updateModelSummary();
}

/* ── Preset Application ──────────────────────────────── */
function applyPreset(preset) {
  // Apply model params
  if (preset.model) {
    const m = preset.model;
    $('#model-name').value = m.name || '';
    $('#hidden-size').value = m.hidden_size || 4096;
    $('#vocab-size').value = m.vocab_size || 32000;
    $('#num-layers').value = m.num_layers || 32;
    $('#dtype').value = m.dtype || 'bf16';

    if (m.layer) {
      $('#attention-type').value = m.layer.attention || 'GQA';
      $('#ffn-type').value = m.layer.ffn || 'SwiGLU';
      $('#residual-type').value = m.layer.residual || 'standard';
      $('#num-heads').value = m.layer.num_heads || 32;
      $('#num-kv-heads').value = m.layer.num_kv_heads || 8;
      $('#head-dim').value = m.layer.head_dim || 128;
      $('#intermediate-size').value = m.layer.intermediate_size || 14336;
      $('#num-experts').value = m.layer.num_experts || 1;
      $('#top-k').value = m.layer.top_k || 1;
      $('#shared-experts').value = m.layer.num_shared_experts || 0;
      $('#expert-intermediate-size').value = m.layer.expert_intermediate_size || 0;
      $('#shared-intermediate-size').value = m.layer.shared_intermediate_size || 0;
      $('#kv-compression-dim').value = m.layer.kv_compression_dim || 0;
      $('#query-compression-dim').value = m.layer.query_compression_dim || 0;
      $('#rope-dim').value = m.layer.rope_dim || 0;
      $('#window-size').value = m.layer.window_size || 0;
      $('#mhc-expansion').value = m.layer.mhc_expansion || 4;
    }
  }

  // Apply hardware
  if (preset.hardware) {
    $('#hw-profile').value = preset.hardware;
    const prof = state.hardware[preset.hardware];
    if (prof) {
      $('#devices-per-node').value = prof.devices_per_node;
    }
  }
  if (preset.total_devices) {
    $('#total-devices').value = preset.total_devices;
  }

  // Apply parallelism
  if (preset.parallelism) {
    const p = preset.parallelism;
    $('#par-tp').value = p.tp || 1;
    $('#par-pp').value = p.pp || 1;
    $('#par-ep').value = p.ep || 1;
    $('#par-cp').value = p.cp || 1;
    $('#cp-type').value = p.cp_type || 'ring';
    $('#par-sp').checked = !!p.sp;
    $('#zero-stage').value = p.zero_stage || 0;
    $('#pp-schedule').value = p.pp_schedule || '1f1b';
    $('#recompute-attn').checked = !!p.recompute_attention;
    $('#full-recompute').checked = !!p.full_recomputation;
    $('#opt-offload').checked = !!p.optimizer_offload;
    $('#act-offload').checked = !!p.activation_offload;
  }

  // Apply RL config
  if (preset.rl) {
    const r = preset.rl;
    $('#total-prompts').value = r.total_prompts || 10000;
    $('#group-size').value = r.group_size || 8;
    $('#total-responses').value = (r.total_prompts || 10000) * (r.group_size || 8);
    $('#avg-prompt-len').value = r.avg_prompt_len || 512;
    $('#avg-response-len').value = r.avg_response_len || 2048;
    $('#max-response-len').value = r.max_response_len || 4096;
    $('#std-response-len').value = r.std_response_len || '';
    $('#train-mbs').value = r.train_micro_batch_size || 4;
    $('#grad-accum').value = r.gradient_accumulation_steps || 1;
    $('#gen-batch-size').value = r.gen_batch_size || 64;
    $('#deploy-mode').value = r.colocated ? 'colocated' : 'separate';
    $('#ref-model').checked = r.reference_model !== false;
    $('#ref-offload').checked = !!r.ref_offload_cpu;
    $('#spec-decode').checked = !!r.use_speculative_decoding;
    if (r.mtp_acceptance_len) {
      $('#mtp-acceptance-len').value = r.mtp_acceptance_len;
    }
    $('#mtp-fields').classList.toggle('hidden', !r.use_speculative_decoding);
  }

  // Recompute derived fields
  recomputeDPFromPreset();
  updateConditionalFields();
  updateModelSummary();
  updateHardwareSummary();
  updateRLSummary();
  resetModified();
  refreshJsonEditor();
}

function recomputeDPFromPreset() {
  const total = intVal('total-devices');
  const tp = intVal('par-tp');
  const pp = intVal('par-pp');
  const ep = intVal('par-ep');
  const divisor = tp * pp * ep;
  const dp = divisor > 0 ? Math.floor(total / divisor) : total;
  $('#par-dp').value = dp;

  const perNode = intVal('devices-per-node');
  const nodes = perNode > 0 ? Math.ceil(total / perNode) : 1;
  $('#num-nodes').value = nodes;
}

function applyConfigToForm(config) {
  // Wrapper: apply a full predict-request-shaped config object to form fields
  if (config.model) {
    // Reshape to preset format
    const preset = {
      model: config.model,
      hardware: config.hardware,
      total_devices: config.total_devices,
      parallelism: config.parallelism,
      rl: config.rl,
    };
    applyPreset(preset);
  }
}

/* ── JSON Config Editor ──────────────────────────────── */
function initJsonEditor() {
  const refreshBtn = $('#json-refresh-btn');
  const applyBtn = $('#json-apply-btn');

  if (refreshBtn) {
    refreshBtn.addEventListener('click', () => {
      refreshJsonEditor();
    });
  }

  if (applyBtn) {
    applyBtn.addEventListener('click', async () => {
      const textarea = $('#json-config-editor');
      const status = $('#json-editor-status');
      const raw = textarea.value.trim();

      if (!raw) return;

      try {
        const config = JSON.parse(raw);
        textarea.classList.remove('json-error');
        textarea.classList.add('json-success');

        applyConfigToForm(config);

        status.classList.remove('hidden', 'error');
        status.classList.add('success');
        status.textContent = 'JSON applied to form fields.';

        // Auto-run after applying JSON
        await runAnalysis();

        // Clear success state after a moment
        setTimeout(() => {
          textarea.classList.remove('json-success');
          status.classList.add('hidden');
        }, 2000);
      } catch (e) {
        textarea.classList.remove('json-success');
        textarea.classList.add('json-error');
        status.classList.remove('hidden', 'success');
        status.classList.add('error');
        status.textContent = `Invalid JSON: ${e.message}`;
      }
    });
  }
}

function refreshJsonEditor() {
  const textarea = $('#json-config-editor');
  if (textarea) {
    const config = buildPredictRequest();
    textarea.value = JSON.stringify(config, null, 2);
    textarea.classList.remove('json-error', 'json-success');
    const status = $('#json-editor-status');
    if (status) status.classList.add('hidden');
  }
}

/* ── Loading / Error States ───────────────────────────── */
function setLoading(loading) {
  const kpis = $$('.kpi-card');
  const chartLoading = $('#chart-loading');

  kpis.forEach((kpi) => {
    if (loading) {
      kpi.classList.add('loading');
      kpi.classList.remove('fade-in');
    } else {
      kpi.classList.remove('loading');
    }
  });

  chartLoading.classList.toggle('hidden', !loading);
}

function showError(msg) {
  const banner = $('#error-banner');
  const text = $('#error-text');
  text.textContent = msg;
  banner.classList.remove('hidden');
}

function hideError() {
  $('#error-banner').classList.add('hidden');
}

function hideEmptyState() {
  const el = $('#empty-state');
  if (el) el.classList.add('hidden');
}

/* ── KPI Rendering ────────────────────────────────────── */
function renderKPIs(kpis, memory) {
  const feasible = kpis.feasible && memory.train_feasible && memory.gen_feasible;
  const withinBudget = kpis.within_budget;

  // Epoch Time
  setKPI('kpi-epoch', {
    value: `${kpis.epoch_time_hours.toFixed(2)}h`,
    detail: feasible
      ? withinBudget ? 'Feasible, within budget' : 'Feasible, over budget'
      : 'Infeasible',
    status: feasible ? (withinBudget ? 'success' : 'warning') : 'error',
  });

  // Gen TPS
  setKPI('kpi-gen', {
    value: formatNumber(kpis.gen_tps_target),
    detail: `gen: ${kpis.gen_time_hours.toFixed(2)}h`,
    status: memory.gen_feasible ? 'success' : 'error',
  });

  // Train TPS
  setKPI('kpi-train', {
    value: formatNumber(kpis.train_tps_target),
    detail: `train: ${kpis.train_time_hours.toFixed(2)}h`,
    status: memory.train_feasible ? 'success' : 'error',
  });

  // Bottleneck
  const slackPct = (kpis.bottleneck_slack * 100).toFixed(1);
  setKPI('kpi-bottleneck', {
    value: capitalize(kpis.bottleneck),
    detail: `${slackPct}% slack`,
    status: 'warning',
  });
}

function renderErrorKPIs() {
  ['kpi-epoch', 'kpi-gen', 'kpi-train', 'kpi-bottleneck'].forEach((id) => {
    setKPI(id, { value: '\u2014', detail: 'error', status: 'error' });
  });
}

function setKPI(id, { value, detail, status }) {
  const card = $(`#${id}`);
  card.querySelector('.kpi-value').textContent = value;
  card.querySelector('.kpi-detail').textContent = detail;
  card.dataset.status = status;
  card.classList.add('fade-in');
}

/* ── Chart: Timeline ──────────────────────────────────── */
function renderTimeline(timeline) {
  const genH = timeline.gen_hours;
  const trainH = timeline.train_hours;
  const colocated = timeline.colocated;

  const traces = [];

  if (colocated) {
    // Overlapping bars
    traces.push({
      y: ['Epoch'],
      x: [Math.max(genH, trainH)],
      type: 'bar',
      orientation: 'h',
      name: 'Total',
      marker: { color: '#e2e8f0' },
    });
    traces.push({
      y: ['Generation'],
      x: [genH],
      type: 'bar',
      orientation: 'h',
      name: 'Generation',
      marker: { color: '#7c3aed' },
    });
    traces.push({
      y: ['Training'],
      x: [trainH],
      type: 'bar',
      orientation: 'h',
      name: 'Training',
      marker: { color: '#ea580c' },
    });
  } else {
    // Sequential: gen then train
    traces.push({
      y: ['Epoch'],
      x: [genH],
      type: 'bar',
      orientation: 'h',
      name: 'Generation',
      marker: { color: '#7c3aed' },
      text: [`${genH.toFixed(2)}h`],
      textposition: 'inside',
      textfont: { color: '#fff', size: 12 },
    });
    traces.push({
      y: ['Epoch'],
      x: [trainH],
      type: 'bar',
      orientation: 'h',
      name: 'Training',
      marker: { color: '#ea580c' },
      text: [`${trainH.toFixed(2)}h`],
      textposition: 'inside',
      textfont: { color: '#fff', size: 12 },
    });
  }

  const layout = {
    barmode: colocated ? 'group' : 'stack',
    ...chartLayout('Timeline (hours)'),
    xaxis: {
      title: 'Hours',
      gridcolor: '#f0eeeb',
      zeroline: false,
    },
    yaxis: {
      automargin: true,
    },
    height: 200,
    margin: { l: 80, r: 30, t: 30, b: 40 },
  };

  Plotly.newPlot('chart-timeline', traces, layout, plotlyConfig());
}

/* ── Chart: Memory ────────────────────────────────────── */
function renderMemory(memory) {
  const categories = ['Per-Device Memory'];

  const traces = [
    {
      y: categories,
      x: [memory.weight_gb],
      name: 'Weights',
      type: 'bar',
      orientation: 'h',
      marker: { color: '#7c3aed' },
    },
    {
      y: categories,
      x: [memory.optimizer_gb],
      name: 'Optimizer',
      type: 'bar',
      orientation: 'h',
      marker: { color: '#c4b5fd' },
    },
    {
      y: categories,
      x: [memory.activation_peak_gb],
      name: 'Activations',
      type: 'bar',
      orientation: 'h',
      marker: { color: '#f59e0b' },
    },
    {
      y: categories,
      x: [memory.ref_model_gb],
      name: 'Ref Model',
      type: 'bar',
      orientation: 'h',
      marker: { color: '#06b6d4' },
    },
    {
      y: categories,
      x: [memory.kv_cache_gb],
      name: 'KV Cache',
      type: 'bar',
      orientation: 'h',
      marker: { color: '#16a34a' },
    },
  ];

  const shapes = [
    {
      type: 'line',
      x0: memory.usable_hbm_gb,
      x1: memory.usable_hbm_gb,
      y0: -0.5,
      y1: 0.5,
      line: { color: '#dc2626', width: 2, dash: 'dash' },
    },
  ];

  const annotations = [
    {
      x: memory.usable_hbm_gb,
      y: 0.55,
      text: `HBM Limit: ${memory.usable_hbm_gb}GB`,
      showarrow: false,
      font: { size: 11, color: '#dc2626' },
      yref: 'paper',
      yanchor: 'bottom',
    },
  ];

  const layout = {
    barmode: 'stack',
    ...chartLayout('Memory Breakdown (GB)'),
    xaxis: {
      title: 'GB',
      gridcolor: '#f0eeeb',
      zeroline: false,
    },
    yaxis: {
      automargin: true,
    },
    shapes,
    annotations,
    height: 220,
    margin: { l: 120, r: 30, t: 30, b: 40 },
  };

  Plotly.newPlot('chart-memory', traces, layout, plotlyConfig());
}

/* ── Chart: Topology ──────────────────────────────────── */
function renderTopology(topology) {
  const ranks = topology.ranks;
  if (!ranks || ranks.length === 0) return;

  const x = ranks.map((r) => r.tp_rank);
  const y = ranks.map((r) => r.pp_stage);
  const text = ranks.map(
    (r) =>
      `Rank ${r.global_rank}<br>TP=${r.tp_rank} PP=${r.pp_stage}<br>DP=${r.dp_rank} EP=${r.ep_rank}<br>Node ${r.node}, GPU ${r.local_gpu}<br>Layers ${r.layer_start}-${r.layer_end}`
  );

  // Color by PP stage
  const colors = ranks.map((r) => r.pp_stage);
  // Opacity: full for dp_rank 0, lower for replicas
  const opacities = ranks.map((r) => (r.dp_rank === 0 ? 1.0 : 0.35));

  // Border by EP group
  const borderColors = ranks.map((r) => {
    const epColors = ['#1a1a1a', '#2563eb', '#16a34a', '#ea580c', '#7c3aed', '#06b6d4'];
    return epColors[r.ep_rank % epColors.length];
  });

  const trace = {
    x,
    y,
    text,
    mode: 'markers',
    type: 'scatter',
    hoverinfo: 'text',
    marker: {
      size: 28,
      color: colors,
      colorscale: 'Viridis',
      opacity: opacities,
      line: {
        width: 2,
        color: borderColors,
      },
    },
  };

  const layout = {
    ...chartLayout('Device Topology'),
    xaxis: {
      title: 'TP Rank',
      dtick: 1,
      gridcolor: '#f0eeeb',
      zeroline: false,
    },
    yaxis: {
      title: 'PP Stage',
      dtick: 1,
      autorange: 'reversed',
      gridcolor: '#f0eeeb',
      zeroline: false,
    },
    height: 400,
    margin: { l: 60, r: 30, t: 30, b: 50 },
  };

  Plotly.newPlot('chart-topology', [trace], layout, plotlyConfig());
}

/* ── Chart: Search Results ────────────────────────────── */
function renderSearchResults(data, mode) {
  const results = data.results || [];
  if (results.length === 0) return;

  if (mode === 'pareto') {
    renderParetoChart(results);
  } else {
    renderSensitivityChart(results);
  }

  renderSearchTable(results, mode);
}

function renderParetoChart(results) {
  const feasible = results.filter((r) => !r.is_oom);
  const pareto = results.filter((r) => r.is_pareto);
  const oom = results.filter((r) => r.is_oom);

  const traces = [];

  if (feasible.length > 0) {
    traces.push({
      x: feasible.map((r) => r.devices),
      y: feasible.map((r) => r.epoch_time_hours),
      mode: 'markers',
      type: 'scatter',
      name: 'Feasible',
      marker: { color: '#6b7280', size: 8, opacity: 0.5 },
      text: feasible.map((r) => `${r.devices} devices<br>TP=${r.parallelism.tp} PP=${r.parallelism.pp}<br>${r.epoch_time_hours.toFixed(2)}h`),
      hoverinfo: 'text',
    });
  }

  if (pareto.length > 0) {
    traces.push({
      x: pareto.map((r) => r.devices),
      y: pareto.map((r) => r.epoch_time_hours),
      mode: 'markers+lines',
      type: 'scatter',
      name: 'Pareto Optimal',
      marker: { color: '#CF0A2C', size: 12, symbol: 'diamond' },
      line: { color: '#CF0A2C', width: 1, dash: 'dot' },
      text: pareto.map((r) => `${r.devices} devices<br>TP=${r.parallelism.tp} PP=${r.parallelism.pp}<br>${r.epoch_time_hours.toFixed(2)}h`),
      hoverinfo: 'text',
    });
  }

  if (oom.length > 0) {
    traces.push({
      x: oom.map((r) => r.devices),
      y: oom.map((r) => r.epoch_time_hours),
      mode: 'markers',
      type: 'scatter',
      name: 'OOM',
      marker: { color: '#dc2626', size: 8, symbol: 'x', opacity: 0.5 },
      text: oom.map((r) => `${r.devices} devices (OOM)`),
      hoverinfo: 'text',
    });
  }

  const layout = {
    ...chartLayout('Pareto Search — Devices vs Epoch Time'),
    xaxis: { title: 'Devices', gridcolor: '#f0eeeb', zeroline: false },
    yaxis: { title: 'Epoch Time (hours)', gridcolor: '#f0eeeb', zeroline: false },
    height: 350,
    margin: { l: 60, r: 30, t: 30, b: 50 },
  };

  Plotly.newPlot('chart-search', traces, layout, plotlyConfig());
}

function renderSensitivityChart(results) {
  const xVals = results.map((r) => String(r.sweep_value));
  const yVals = results.map((r) => r.epoch_time_hours);
  const colors = results.map((r) => (r.is_oom ? '#dc2626' : r.feasible ? '#16a34a' : '#f59e0b'));

  const trace = {
    x: xVals,
    y: yVals,
    type: 'bar',
    marker: { color: colors },
    text: yVals.map((v) => `${v.toFixed(2)}h`),
    textposition: 'outside',
    textfont: { size: 11 },
  };

  const layout = {
    ...chartLayout('Sensitivity Analysis'),
    xaxis: {
      title: $('#sweep-param').value,
      gridcolor: '#f0eeeb',
      zeroline: false,
      type: 'category',
    },
    yaxis: {
      title: 'Epoch Time (hours)',
      gridcolor: '#f0eeeb',
      zeroline: false,
    },
    height: 350,
    margin: { l: 60, r: 30, t: 30, b: 50 },
  };

  Plotly.newPlot('chart-search', [trace], layout, plotlyConfig());
}

function renderSearchTable(results, mode) {
  const wrap = $('#search-table-wrap');
  const table = $('#search-table');
  const thead = table.querySelector('thead');
  const tbody = table.querySelector('tbody');

  wrap.classList.remove('hidden');

  if (mode === 'pareto') {
    thead.innerHTML = `<tr>
      <th>Devices</th><th>TP</th><th>PP</th><th>DP</th><th>EP</th>
      <th>Epoch (h)</th><th>Gen TPS</th><th>Train TPS</th><th>Status</th>
    </tr>`;
    tbody.innerHTML = results
      .map((r) => {
        const cls = r.is_oom ? 'oom' : r.is_pareto ? 'pareto' : '';
        const statusText = r.is_oom ? 'OOM' : r.feasible ? (r.is_pareto ? 'Pareto' : 'OK') : 'Infeasible';
        return `<tr class="${cls}">
          <td>${r.devices}</td><td>${r.parallelism.tp}</td><td>${r.parallelism.pp}</td>
          <td>${r.parallelism.dp}</td><td>${r.parallelism.ep}</td>
          <td>${r.epoch_time_hours.toFixed(2)}</td><td>${formatNumber(r.gen_tps)}</td>
          <td>${formatNumber(r.train_tps)}</td><td>${statusText}</td>
        </tr>`;
      })
      .join('');
  } else {
    thead.innerHTML = `<tr>
      <th>Value</th><th>Epoch (h)</th><th>Gen TPS</th><th>Train TPS</th><th>Status</th>
    </tr>`;
    tbody.innerHTML = results
      .map((r) => {
        const cls = r.is_oom ? 'oom' : '';
        const statusText = r.is_oom ? 'OOM' : r.feasible ? 'OK' : 'Infeasible';
        return `<tr class="${cls}">
          <td>${r.sweep_value}</td>
          <td>${r.epoch_time_hours.toFixed(2)}</td><td>${formatNumber(r.gen_tps)}</td>
          <td>${formatNumber(r.train_tps)}</td><td>${statusText}</td>
        </tr>`;
      })
      .join('');
  }
}

/* ── Plotly Helpers ────────────────────────────────────── */
function chartLayout(title) {
  return {
    title: { text: '', font: { size: 14 } },
    paper_bgcolor: '#FAFAF8',
    plot_bgcolor: '#FAFAF8',
    font: {
      family: 'DM Sans, system-ui, sans-serif',
      size: 12,
      color: '#1a1a1a',
    },
    showlegend: true,
    legend: {
      orientation: 'h',
      y: -0.25,
      x: 0,
      font: { size: 11 },
    },
  };
}

function plotlyConfig() {
  return {
    responsive: true,
    displayModeBar: false,
  };
}

/* ── Utility ──────────────────────────────────────────── */
function intVal(id) {
  return parseInt($(`#${id}`).value, 10) || 0;
}

function intValOrNull(id) {
  const v = $(`#${id}`).value;
  if (v === '' || v === null || v === undefined) return null;
  const n = parseInt(v, 10);
  return isNaN(n) ? null : n;
}

function formatNumber(n) {
  if (n === null || n === undefined) return '\u2014';
  return n.toLocaleString('en-US', { maximumFractionDigits: 0 });
}

function capitalize(s) {
  if (!s) return '';
  return s.charAt(0).toUpperCase() + s.slice(1);
}

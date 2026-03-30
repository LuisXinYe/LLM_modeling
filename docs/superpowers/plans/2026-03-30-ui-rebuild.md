# rl-perf UI Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Gradio GUI with a FastAPI + vanilla HTML/CSS/JS single-page app that gives full design control per Impeccable `/frontend-design` guidelines.

**Architecture:** FastAPI serves static files (index.html, styles.css, app.js) and 5 REST endpoints (`/api/models`, `/api/hardware`, `/api/predict`, `/api/search`, `/api/hf-import`). Browser renders charts client-side with Plotly.js CDN. No build step, no npm, stays pure Python.

**Tech Stack:** Python 3.10+, FastAPI, uvicorn, Plotly.js (CDN), vanilla HTML/CSS/JS

**Design spec:** `docs/superpowers/specs/2026-03-30-ui-rebuild-design.md`
**Design context:** `.impeccable.md`

---

## File Structure

```
src/rl_perf/ui/
├── __init__.py          # (keep, empty)
├── api.py               # NEW: FastAPI app, REST endpoints, static serving
├── hf_import.py         # KEEP: HuggingFace import logic (used by api.py)
└── static/
    ├── index.html       # NEW: Single-page app structure
    ├── styles.css        # NEW: Design tokens + all component styles
    └── app.js            # NEW: Accordion, tabs, fetch, Plotly chart rendering
```

**Delete:** `app.py`, `tab_model.py`, `tab_hardware.py`, `tab_rl.py`, `tab_search.py`, `results.py`, `plots.py`, `topology.py`, `_theme.py`, `ai_sidebar.py`

**Modify:** `cli.py` (update `ui` command), `pyproject.toml` (swap gradio for fastapi)

---

### Task 1: Delete old Gradio UI and update dependencies

**Files:**
- Delete: `src/rl_perf/ui/app.py`, `src/rl_perf/ui/tab_model.py`, `src/rl_perf/ui/tab_hardware.py`, `src/rl_perf/ui/tab_rl.py`, `src/rl_perf/ui/tab_search.py`, `src/rl_perf/ui/results.py`, `src/rl_perf/ui/plots.py`, `src/rl_perf/ui/topology.py`, `src/rl_perf/ui/_theme.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Delete all Gradio UI files**

```bash
cd /Users/horacehxw/Projects/RL_modeling/.worktrees/ui_optimize
git rm src/rl_perf/ui/app.py src/rl_perf/ui/tab_model.py src/rl_perf/ui/tab_hardware.py src/rl_perf/ui/tab_rl.py src/rl_perf/ui/tab_search.py src/rl_perf/ui/results.py src/rl_perf/ui/plots.py src/rl_perf/ui/topology.py src/rl_perf/ui/_theme.py
```

Keep `hf_import.py` and `__init__.py`.

- [ ] **Step 2: Update pyproject.toml dependencies**

Replace `gradio>=5.0` with `fastapi>=0.100` and `uvicorn[standard]>=0.20` in both `gui` and `dev` optional deps. Keep `plotly>=5.0` (used by tests) and `huggingface-hub>=0.20`.

```toml
[project.optional-dependencies]
gui = ["fastapi>=0.100", "uvicorn[standard]>=0.20", "plotly>=5.0", "huggingface-hub>=0.20"]
dev = ["pytest", "pytest-xdist", "ruff", "fastapi>=0.100", "uvicorn[standard]>=0.20", "plotly>=5.0", "huggingface-hub>=0.20"]
```

- [ ] **Step 3: Install new dependencies**

```bash
source /Users/horacehxw/Projects/RL_modeling/.venv/bin/activate
pip install -e ".[dev]"
```

- [ ] **Step 4: Verify backend tests still pass**

```bash
pytest tests/ -x -q --ignore=tests/test_ui.py 2>/dev/null; pytest tests/ -x -q
```

Expected: All non-UI tests pass. UI tests may fail (expected — we deleted the files).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: remove Gradio UI, switch to FastAPI + uvicorn deps"
```

---

### Task 2: Create FastAPI backend (api.py)

**Files:**
- Create: `src/rl_perf/ui/api.py`
- Modify: `src/rl_perf/cli.py`

- [ ] **Step 1: Create api.py with FastAPI app and all endpoints**

Create `src/rl_perf/ui/api.py`:

```python
"""FastAPI backend for rl-perf web GUI.

Serves static files (index.html, styles.css, app.js) and provides REST
endpoints for model prediction, search, and configuration loading.
"""

from __future__ import annotations

import html
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rl_perf.config import (
    HardwareConfig,
    LayerConfig,
    ModelConfig,
    ParallelismConfig,
    RLConfig,
    load_hardware_config,
    load_model_config,
)
from rl_perf.model import RLPerformanceModel
from rl_perf.search import pareto_search, sensitivity_sweep
from rl_perf.ui.hf_import import fetch_hf_config, hf_config_to_model_config

_STATIC_DIR = Path(__file__).parent / "static"
_CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "configs"

_MODEL_TEMPLATES = {
    "Llama-3.1-8B": "llama3_1_8b",
    "Qwen2.5-72B": "qwen2_5_72b",
    "Mistral-7B": "mistral_7b",
    "Qwen3-235B-MoE": "qwen3_235b_moe",
    "DeepSeekV3-671B": "deepseekv3_671b",
}

_HW_TEMPLATES = {
    "Ascend 910C": "ascend_910c",
    "CloudMatrix 384": "cloudmatrix_384",
}

app = FastAPI(title="rl-perf", docs_url=None, redoc_url=None)


# ── Pydantic models for request/response ──────────────────────────


class LayerInput(BaseModel):
    attention: str = "GQA"
    num_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    ffn: str = "SwiGLU"
    intermediate_size: int = 14336
    residual: str = "standard"
    num_experts: int = 1
    top_k: int = 1
    num_shared_experts: int = 0
    expert_intermediate_size: int = 0
    shared_intermediate_size: int = 0
    kv_compression_dim: int = 0
    query_compression_dim: int = 0
    rope_dim: int = 0
    window_size: int = 0
    mhc_expansion: int = 4


class ModelInput(BaseModel):
    name: str = "Llama-3.1-8B"
    hidden_size: int = 4096
    vocab_size: int = 128256
    num_layers: int = 32
    dtype: str = "bf16"
    layer: LayerInput = LayerInput()


class ParallelismInput(BaseModel):
    tp: int = 1
    pp: int = 1
    dp: int = 8
    ep: int = 1
    cp: int = 1
    cp_type: str = "ring"
    sp: bool = False
    zero_stage: int = 0
    pp_schedule: str = "1f1b"
    recompute_attention: bool = False
    full_recomputation: bool = False
    optimizer_offload: bool = False
    activation_offload: bool = False


class RLInput(BaseModel):
    total_prompts: int = 10000
    group_size: int = 8
    avg_prompt_len: int = 512
    avg_response_len: int = 2048
    max_response_len: int = 4096
    std_response_len: int | None = None
    train_micro_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    gen_batch_size: int = 64
    colocated: bool = False
    reference_model: bool = True
    ref_offload_cpu: bool = False
    use_speculative_decoding: bool = False
    mtp_acceptance_len: int | None = None


class PredictRequest(BaseModel):
    model: ModelInput = ModelInput()
    hardware: str = "Ascend 910C"
    total_devices: int = 8
    parallelism: ParallelismInput = ParallelismInput()
    rl: RLInput = RLInput()


class SearchConfig(BaseModel):
    mode: str = "pareto"
    device_counts: list[int] = [8, 16, 32, 64, 128]
    optimization_target: str = "epoch_time_hours"
    sweep_param: str = "group_size"
    sweep_values: list[int] = [4, 8, 16, 32]


class SearchRequest(BaseModel):
    model: ModelInput = ModelInput()
    hardware: str = "Ascend 910C"
    total_devices: int = 8
    parallelism: ParallelismInput = ParallelismInput()
    rl: RLInput = RLInput()
    search: SearchConfig = SearchConfig()


class HFImportRequest(BaseModel):
    model_id: str


# ── Helpers ───────────────────────────────────────────────────────


def _build_model_config(m: ModelInput) -> ModelConfig:
    layer = LayerConfig(
        attention=m.layer.attention,
        num_heads=m.layer.num_heads,
        num_kv_heads=m.layer.num_kv_heads,
        head_dim=m.layer.head_dim,
        ffn=m.layer.ffn,
        intermediate_size=m.layer.intermediate_size,
        residual=m.layer.residual,
        num_experts=m.layer.num_experts,
        top_k=m.layer.top_k,
        num_shared_experts=m.layer.num_shared_experts,
        expert_intermediate_size=m.layer.expert_intermediate_size,
        shared_intermediate_size=m.layer.shared_intermediate_size,
        kv_compression_dim=m.layer.kv_compression_dim,
        query_compression_dim=m.layer.query_compression_dim,
        rope_dim=m.layer.rope_dim,
        window_size=m.layer.window_size,
        mhc_expansion=m.layer.mhc_expansion,
    )
    return ModelConfig(
        name=m.name,
        hidden_size=m.hidden_size,
        vocab_size=m.vocab_size,
        num_layers=m.num_layers,
        dtype=m.dtype,
        default_layer=layer,
    )


def _build_hw_config(hw_name: str) -> HardwareConfig:
    stem = _HW_TEMPLATES.get(hw_name)
    if not stem:
        raise HTTPException(status_code=400, detail=f"Unknown hardware: {hw_name}")
    return load_hardware_config(str(_CONFIGS_DIR / "hardware" / f"{stem}.yaml"))


def _build_parallelism(p: ParallelismInput) -> ParallelismConfig:
    return ParallelismConfig(
        tp=p.tp, pp=p.pp, dp=p.dp, ep=p.ep, cp=p.cp,
        cp_type=p.cp_type, sp=p.sp,
        zero_stage=p.zero_stage, pp_schedule=p.pp_schedule,
        recompute_attention=p.recompute_attention,
        full_recomputation=p.full_recomputation,
        optimizer_offload=p.optimizer_offload,
        activation_offload=p.activation_offload,
    )


def _build_rl_config(r: RLInput) -> RLConfig:
    std = r.std_response_len if r.std_response_len and r.std_response_len > 0 else None
    mtp = r.mtp_acceptance_len if r.use_speculative_decoding else None
    return RLConfig(
        total_prompts=r.total_prompts,
        group_size=r.group_size,
        avg_prompt_len=r.avg_prompt_len,
        avg_response_len=r.avg_response_len,
        max_response_len=r.max_response_len,
        std_response_len=std,
        train_micro_batch_size=r.train_micro_batch_size,
        gradient_accumulation_steps=r.gradient_accumulation_steps,
        gen_batch_size=r.gen_batch_size,
        colocated=r.colocated,
        reference_model=r.reference_model,
        ref_offload_cpu=r.ref_offload_cpu,
        use_speculative_decoding=r.use_speculative_decoding,
        mtp_acceptance_len=mtp,
    )


def _topology_data(par: ParallelismInput, hw: HardwareConfig, num_layers: int) -> list[dict]:
    """Compute rank mapping for topology visualization."""
    tp, ep, pp, dp = par.tp, par.ep, par.pp, par.dp
    total = tp * ep * pp * dp
    layers_per_stage = num_layers // pp if pp > 0 else num_layers
    ranks = []
    for g in range(total):
        r = g
        tp_rank = r % tp; r //= tp
        ep_rank = r % ep; r //= ep
        pp_stage = r % pp; r //= pp
        dp_rank = r
        ranks.append({
            "global_rank": g,
            "node": g // hw.devices_per_node,
            "local_gpu": g % hw.devices_per_node,
            "tp_rank": tp_rank,
            "pp_stage": pp_stage,
            "dp_rank": dp_rank,
            "ep_rank": ep_rank,
            "layer_start": pp_stage * layers_per_stage,
            "layer_end": pp_stage * layers_per_stage + layers_per_stage - 1,
        })
    return ranks


# ── Endpoints ─────────────────────────────────────────────────────


@app.get("/api/models")
def get_models():
    templates = {}
    for display_name, stem in _MODEL_TEMPLATES.items():
        yaml_path = _CONFIGS_DIR / "models" / f"{stem}.yaml"
        if yaml_path.exists():
            mc = load_model_config(str(yaml_path))
            layer = mc.default_layer or LayerConfig()
            templates[display_name] = {
                "name": mc.name,
                "hidden_size": mc.hidden_size,
                "vocab_size": mc.vocab_size,
                "num_layers": mc.num_layers,
                "dtype": mc.dtype,
                "layer": {
                    "attention": layer.attention,
                    "num_heads": layer.num_heads,
                    "num_kv_heads": layer.num_kv_heads,
                    "head_dim": layer.head_dim,
                    "ffn": layer.ffn,
                    "intermediate_size": layer.intermediate_size,
                    "residual": layer.residual,
                    "num_experts": layer.num_experts,
                    "top_k": layer.top_k,
                    "num_shared_experts": layer.num_shared_experts,
                    "expert_intermediate_size": layer.expert_intermediate_size,
                    "shared_intermediate_size": layer.shared_intermediate_size,
                    "kv_compression_dim": layer.kv_compression_dim,
                    "query_compression_dim": layer.query_compression_dim,
                    "rope_dim": layer.rope_dim,
                    "window_size": layer.window_size,
                    "mhc_expansion": layer.mhc_expansion,
                },
            }
    return {"templates": templates}


@app.get("/api/hardware")
def get_hardware():
    profiles = {}
    for display_name, stem in _HW_TEMPLATES.items():
        yaml_path = _CONFIGS_DIR / "hardware" / f"{stem}.yaml"
        if yaml_path.exists():
            hw = load_hardware_config(str(yaml_path))
            profiles[display_name] = {
                "devices_per_node": hw.devices_per_node,
                "hbm_gb": hw.hbm_capacity_gb,
                "tflops_bf16": hw.peak_tflops_bf16,
            }
    return {"profiles": profiles}


@app.post("/api/predict")
def predict(req: PredictRequest):
    try:
        model_cfg = _build_model_config(req.model)
        hw_cfg = _build_hw_config(req.hardware)
        train_par = _build_parallelism(req.parallelism)
        rl_cfg = _build_rl_config(req.rl)

        gen_dp = req.total_devices // req.parallelism.tp if req.parallelism.tp > 0 else 1
        gen_par = ParallelismConfig(tp=req.parallelism.tp, pp=1, dp=gen_dp)

        perf = RLPerformanceModel(model_cfg, hw_cfg)
        report = perf.derive_targets(req.total_devices, rl_cfg, gen_par, train_par)
        mem = report.memory

        topo = _topology_data(req.parallelism, hw_cfg, model_cfg.num_layers)

        return {
            "kpis": {
                "epoch_time_hours": round(report.epoch_time_hours, 4),
                "gen_tps_target": round(report.gen_tps_target, 0),
                "train_tps_target": round(report.train_tps_target, 0),
                "gen_time_hours": round(report.gen_time_hours, 4),
                "train_time_hours": round(report.train_time_hours, 4),
                "bottleneck": report.bottleneck,
                "bottleneck_slack": round(report.bottleneck_slack, 4),
                "feasible": report.feasible,
                "within_budget": report.within_budget,
            },
            "memory": {
                "weight_gb": round(mem.weight_gb, 2),
                "optimizer_gb": round(mem.optimizer_gb, 2),
                "activation_peak_gb": round(mem.activation_peak_gb, 2),
                "ref_model_gb": round(mem.ref_model_gb, 2),
                "kv_cache_gb": round(mem.kv_cache_gb, 2),
                "usable_hbm_gb": round(mem.usable_hbm_gb, 2),
                "train_feasible": mem.train_feasible,
                "gen_feasible": mem.gen_feasible,
            },
            "timeline": {
                "gen_hours": round(report.gen_time_hours, 4),
                "train_hours": round(report.train_time_hours, 4),
                "colocated": req.rl.colocated,
            },
            "topology": {
                "ranks": topo,
                "tp": req.parallelism.tp,
                "pp": req.parallelism.pp,
                "dp": req.parallelism.dp,
                "ep": req.parallelism.ep,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=422, detail=html.escape(str(e)))


@app.post("/api/search")
def search(req: SearchRequest):
    try:
        model_cfg = _build_model_config(req.model)
        hw_cfg = _build_hw_config(req.hardware)
        rl_cfg = _build_rl_config(req.rl)
        perf = RLPerformanceModel(model_cfg, hw_cfg)

        if req.search.mode == "pareto":
            sr = pareto_search(perf, hw_cfg, rl_cfg, req.search.device_counts)
            results = []
            for r in sr:
                tp_cfg = r.train_parallel
                results.append({
                    "devices": r.devices,
                    "parallelism": {"tp": tp_cfg.tp, "pp": tp_cfg.pp, "dp": tp_cfg.dp, "ep": tp_cfg.ep},
                    "epoch_time_hours": round(r.report.epoch_time_hours, 4),
                    "gen_tps": round(r.report.gen_tps_target, 0),
                    "train_tps": round(r.report.train_tps_target, 0),
                    "feasible": r.is_feasible,
                    "is_pareto": r.is_pareto,
                    "is_oom": r.is_oom,
                })
            return {"results": results, "status": f"Pareto search complete. {len(sr)} configs evaluated."}

        else:
            tp_v = req.parallelism.tp
            train_par = _build_parallelism(req.parallelism)
            gen_dp = req.total_devices // tp_v if tp_v > 0 else 1
            gen_par = ParallelismConfig(tp=tp_v, pp=1, dp=gen_dp)

            sweep = sensitivity_sweep(
                perf, hw_cfg, rl_cfg,
                param_name=req.search.sweep_param,
                values=req.search.sweep_values,
                total_devices=req.total_devices,
                gen_parallel=gen_par,
                train_parallel=train_par,
            )
            results = []
            for val, sr in zip(req.search.sweep_values, sweep):
                results.append({
                    "devices": sr.devices,
                    "parallelism": {"tp": train_par.tp, "pp": train_par.pp, "dp": train_par.dp, "ep": train_par.ep},
                    "epoch_time_hours": round(sr.report.epoch_time_hours, 4),
                    "gen_tps": round(sr.report.gen_tps_target, 0),
                    "train_tps": round(sr.report.train_tps_target, 0),
                    "feasible": sr.is_feasible,
                    "is_pareto": False,
                    "is_oom": sr.is_oom,
                    "sweep_value": val,
                })
            return {"results": results, "status": f"Sensitivity sweep complete. {len(sweep)} values evaluated."}

    except Exception as e:
        raise HTTPException(status_code=422, detail=html.escape(str(e)))


@app.post("/api/hf-import")
def hf_import(req: HFImportRequest):
    try:
        hf_cfg = fetch_hf_config(req.model_id)
        mc = hf_config_to_model_config(hf_cfg, name=req.model_id)
        layer = mc.default_layer or LayerConfig()
        return {
            "name": mc.name,
            "hidden_size": mc.hidden_size,
            "vocab_size": mc.vocab_size,
            "num_layers": mc.num_layers,
            "dtype": mc.dtype,
            "layer": {
                "attention": layer.attention,
                "num_heads": layer.num_heads,
                "num_kv_heads": layer.num_kv_heads,
                "head_dim": layer.head_dim,
                "ffn": layer.ffn,
                "intermediate_size": layer.intermediate_size,
                "residual": layer.residual,
                "num_experts": layer.num_experts,
                "top_k": layer.top_k,
                "num_shared_experts": layer.num_shared_experts,
                "expert_intermediate_size": layer.expert_intermediate_size,
                "shared_intermediate_size": layer.shared_intermediate_size,
                "kv_compression_dim": layer.kv_compression_dim,
                "query_compression_dim": layer.query_compression_dim,
                "rope_dim": layer.rope_dim,
                "window_size": layer.window_size,
                "mhc_expansion": layer.mhc_expansion,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=422, detail=html.escape(str(e)))


# ── Static files & SPA fallback ───────────────────────────────────

app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(str(_STATIC_DIR / "index.html"))


def launch(host: str = "127.0.0.1", port: int = 7860):
    """Launch the web GUI."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)
```

- [ ] **Step 2: Update cli.py to use FastAPI launcher**

In `src/rl_perf/cli.py`, change the `ui` command:

```python
@app.command()
def ui(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host (0.0.0.0 for LAN)"),
    port: int = typer.Option(7860, "--port", help="Port number"),
):
    """Launch the web GUI."""
    from rl_perf.ui.api import launch
    launch(host=host, port=port)
```

Remove the `share` parameter (not applicable to FastAPI).

- [ ] **Step 3: Create empty static directory with placeholder**

```bash
mkdir -p src/rl_perf/ui/static
echo "<h1>rl-perf</h1><p>UI loading...</p>" > src/rl_perf/ui/static/index.html
touch src/rl_perf/ui/static/styles.css
touch src/rl_perf/ui/static/app.js
```

- [ ] **Step 4: Verify API starts and endpoints respond**

```bash
source /Users/horacehxw/Projects/RL_modeling/.venv/bin/activate
python -c "
from rl_perf.ui.api import app
from fastapi.testclient import TestClient
c = TestClient(app)
assert c.get('/').status_code == 200
assert c.get('/api/models').status_code == 200
assert c.get('/api/hardware').status_code == 200
r = c.post('/api/predict', json={})
assert r.status_code == 200
print('All endpoints OK')
print('Predict result keys:', list(r.json().keys()))
"
```

Expected: `All endpoints OK` and keys `['kpis', 'memory', 'timeline', 'topology']`.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(gui): FastAPI backend with predict, search, hf-import endpoints"
```

---

### Task 3: Create HTML structure (index.html)

**Files:**
- Create: `src/rl_perf/ui/static/index.html`

- [ ] **Step 1: Write index.html**

Create the complete single-page HTML structure with:
- `<head>`: Google Fonts `<link>` tag (choose a distinctive sans-serif, NOT Inter/Roboto/Open Sans — e.g., DM Sans, Plus Jakarta Sans, or Outfit), Plotly.js CDN `<script>`, link to styles.css
- `<header>`: rl-perf branding with subtitle
- `<main>` with two panels:
  - `<aside id="config-panel">`: 4 accordion sections (Model, Hardware, RL Training, Search) with all form fields per the spec. Each section has a `.accordion-header` (clickable) and `.accordion-body` (collapsible). All inputs have `id` attributes matching the API field names for easy JS binding. Conditional fields (MLA, SWA, MoE, mHC) wrapped in divs with `data-show-when` attributes. Sticky "Run Analysis" button at bottom.
  - `<section id="results-panel">`: Empty state message, KPI cards container (4 divs), tab bar (Timeline, Memory, Topology, Search Results), chart container div.
- `<script src="/static/app.js"></script>` at bottom

Follow the design spec for all field names, types, values, and layout structure. Use semantic HTML — `<form>`, `<fieldset>`, `<label>`, `<select>`, `<input type="number">`, `<input type="checkbox">`.

The HTML should be a complete, self-contained document. Every form field from the spec must be present with its default value.

- [ ] **Step 2: Verify HTML loads in browser**

```bash
source /Users/horacehxw/Projects/RL_modeling/.venv/bin/activate
python -c "
from rl_perf.ui.api import app
from fastapi.testclient import TestClient
c = TestClient(app)
r = c.get('/')
assert r.status_code == 200
assert 'rl-perf' in r.text
assert 'config-panel' in r.text
assert 'results-panel' in r.text
print('HTML structure OK')
"
```

- [ ] **Step 3: Commit**

```bash
git add src/rl_perf/ui/static/index.html
git commit -m "feat(gui): HTML structure with config panel and results layout"
```

---

### Task 4: Create CSS design system (styles.css)

**Files:**
- Create: `src/rl_perf/ui/static/styles.css`

- [ ] **Step 1: Write styles.css**

Create the complete stylesheet following `.impeccable.md` and the Impeccable `/frontend-design` DON'T list. Must include:

**Design tokens** (CSS custom properties in `:root`):
- Typography: `--font-sans`, `--text-xs` through `--text-3xl`, `--weight-normal/medium/semibold/bold`
- Colors: all tokens from the spec (`--accent`, `--bg-page`, `--status-success`, etc.)
- Spacing: `--space-xs` through `--space-2xl`
- Borders: `--radius-sm/md/lg`, `--border-subtle`

**Layout**: Side-by-side with `display: grid; grid-template-columns: 380px 1fr;`. Header above, full-width.

**Config panel styles**: Accordion sections with `grid-template-rows: 0fr` → `1fr` transition for smooth open/close. Accordion headers, form fields (inputs, selects, checkboxes), section sub-headers, sticky run button.

**Results panel styles**: KPI card grid (4 columns), left-border color coding (`.kpi-success`, `.kpi-error`, `.kpi-warning`), tab bar, chart container, empty state, loading skeleton, error banner.

**Form element styles**: Consistent input/select/checkbox styling. Labels using `--text-secondary`. Focus rings with `--accent-light`.

**Responsive**: `@media (max-width: 1024px)` — config panel becomes slide-out drawer. `@media (max-width: 480px)` — KPI cards 2×2.

**Motion**: Accordion transition (250ms), tab crossfade (200ms), button hover (150ms), `@media (prefers-reduced-motion)` override.

**Impeccable anti-patterns AVOIDED**: No glassmorphism, no gradient text, no bounce easing, no cards-in-cards, no pure #000/#fff, no hero metrics, no centered-everything.

- [ ] **Step 2: Commit**

```bash
git add src/rl_perf/ui/static/styles.css
git commit -m "feat(gui): CSS design system with tokens, layout, and components"
```

---

### Task 5: Create JavaScript application logic (app.js)

**Files:**
- Create: `src/rl_perf/ui/static/app.js`

- [ ] **Step 1: Write app.js**

Create the complete vanilla JS application. Must include:

**Initialization** (`DOMContentLoaded`):
- Fetch `/api/models` and `/api/hardware` to populate dropdowns
- Set up accordion click handlers (only one open at a time)
- Set up tab click handlers
- Set up conditional field visibility (MLA/SWA/MoE/mHC)
- Set up auto-computed fields (DP = total / (TP * PP * EP), Nodes = total / per_node, Total Responses = prompts * group)
- Set up "Run Analysis" button click handler
- Set up model source toggle (Template/HuggingFace/Custom)
- Set up HF import "Load" button

**Accordion logic**:
- Click header → toggle `data-open` attribute → CSS transition handles animation
- Update collapsed summary text from current field values
- Status dot: green by default, amber when values changed since last run

**Form data collection** (`collectConfig()`):
- Read all form fields into a JSON object matching the `PredictRequest` schema
- Handle type conversion (string inputs → numbers, checkboxes → booleans)
- Handle null values (std_response_len = 0 → null, mtp_acceptance_len when not speculative)

**Run Analysis** (`runPrediction()`):
- Show loading state (button spinner, skeleton KPIs, chart spinner)
- `fetch('/api/predict', {method: 'POST', body: JSON.stringify(config)})`
- On success: render KPI cards, render active chart tab
- On error: show error banner with escaped message
- Reset status dots to green

**Chart rendering** (using Plotly.js):
- `renderTimeline(data)`: Horizontal bar chart — generation + training phases. Colors: `--data-purple` for gen, `--data-orange` for train. Warm chart background `#FAFAF8`.
- `renderMemory(data)`: Stacked bar — weights, optimizer, activations, ref model, KV cache. Dashed red HBM limit line. Same warm background.
- `renderTopology(data)`: 2D scatter mesh — x=TP rank, y=PP stage (reversed). Color by PP stage. Border by EP group. Opacity 0.5 for DP groups beyond first.
- All charts use consistent font (`font: {family: "DM Sans, system-ui, sans-serif", size: 13}`), warm background, subtle gridlines.

**Search** (`runSearch()`):
- Collect config + search params
- `fetch('/api/search', ...)`
- Render Pareto scatter or sensitivity bars in the Search Results tab
- Render comparison table below chart
- Show the Search Results tab

**Tab switching**:
- Click tab → show corresponding chart container, hide others
- Re-render chart on tab switch (Plotly needs visible container for correct sizing)

**HF Import**:
- Read model ID from text input
- `fetch('/api/hf-import', {method: 'POST', body: JSON.stringify({model_id})})`
- On success: populate all model form fields with returned values
- On error: show inline error message

- [ ] **Step 2: Verify full round-trip works**

```bash
source /Users/horacehxw/Projects/RL_modeling/.venv/bin/activate
python -c "
from rl_perf.ui.api import app
from fastapi.testclient import TestClient
c = TestClient(app)
# Check static files serve
assert c.get('/').status_code == 200
assert c.get('/static/styles.css').status_code == 200
assert c.get('/static/app.js').status_code == 200
# Check prediction still works
r = c.post('/api/predict', json={})
assert r.status_code == 200
assert 'kpis' in r.json()
print('Full stack OK')
"
```

- [ ] **Step 3: Commit**

```bash
git add src/rl_perf/ui/static/app.js
git commit -m "feat(gui): JS application — accordion, tabs, fetch API, Plotly charts"
```

---

### Task 6: Visual polish and Impeccable audit

**Files:**
- Modify: `src/rl_perf/ui/static/index.html`, `src/rl_perf/ui/static/styles.css`, `src/rl_perf/ui/static/app.js`

- [ ] **Step 1: Launch the app and visually inspect**

```bash
source /Users/horacehxw/Projects/RL_modeling/.venv/bin/activate
python -c "from rl_perf.ui.api import launch; launch(port=7862)"
```

Open http://localhost:7862 in browser. Check:
- Header looks branded (not generic)
- Accordion opens/closes smoothly
- All form fields are functional and properly labeled
- "Run Analysis" produces results with KPI cards and charts
- Charts render with warm backgrounds, correct colors, consistent fonts
- Tab switching works
- Loading states work (skeleton KPIs, button spinner)
- Error states work (try invalid config)
- Responsive: resize to <1024px, verify drawer behavior

- [ ] **Step 2: Run Impeccable audit checklist**

Check against the 5 audit dimensions:
1. **Accessibility**: ARIA landmarks on header/main/aside, focus indicators on inputs, semantic HTML
2. **Performance**: No unnecessary re-renders, Plotly CDN cached, no layout thrashing
3. **Theming**: All colors via CSS tokens, no hardcoded hex in JS/HTML
4. **Responsive**: Config drawer on mobile, KPI 2×2 on small screens
5. **Anti-patterns**: No AI slop tells per Impeccable DON'T list

Fix any issues found.

- [ ] **Step 3: Fix visual issues**

Apply fixes for any issues found in the visual inspection and audit.

- [ ] **Step 4: Run backend tests**

```bash
pytest tests/ -x -q
```

Expected: All tests pass (UI tests may need updating or removal).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(gui): visual polish and Impeccable audit fixes"
```

---

### Task 7: Update tests and final verification

**Files:**
- Modify or delete: `tests/test_ui*.py` (if any exist)
- Create: `tests/test_api.py`

- [ ] **Step 1: Create API tests**

Create `tests/test_api.py`:

```python
"""Tests for the FastAPI REST API."""

import pytest
from fastapi.testclient import TestClient
from rl_perf.ui.api import app

client = TestClient(app)


def test_index_returns_html():
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "rl-perf" in r.text


def test_get_models():
    r = client.get("/api/models")
    assert r.status_code == 200
    templates = r.json()["templates"]
    assert "Llama-3.1-8B" in templates
    assert templates["Llama-3.1-8B"]["hidden_size"] == 4096


def test_get_hardware():
    r = client.get("/api/hardware")
    assert r.status_code == 200
    profiles = r.json()["profiles"]
    assert "Ascend 910C" in profiles
    assert profiles["Ascend 910C"]["devices_per_node"] == 8


def test_predict_default():
    r = client.post("/api/predict", json={})
    assert r.status_code == 200
    data = r.json()
    assert "kpis" in data
    assert "memory" in data
    assert "timeline" in data
    assert "topology" in data
    assert data["kpis"]["epoch_time_hours"] > 0
    assert data["kpis"]["bottleneck"] in ("generation", "training")


def test_predict_custom_config():
    r = client.post("/api/predict", json={
        "model": {"name": "test", "hidden_size": 4096, "vocab_size": 32000, "num_layers": 16, "dtype": "bf16"},
        "hardware": "Ascend 910C",
        "total_devices": 16,
        "parallelism": {"tp": 2, "pp": 1, "dp": 8, "ep": 1, "cp": 1},
    })
    assert r.status_code == 200
    assert r.json()["kpis"]["epoch_time_hours"] > 0


def test_predict_invalid_hardware():
    r = client.post("/api/predict", json={"hardware": "NonExistent"})
    assert r.status_code == 400 or r.status_code == 422


def test_search_pareto():
    r = client.post("/api/search", json={
        "search": {"mode": "pareto", "device_counts": [8, 16]},
    })
    assert r.status_code == 200
    data = r.json()
    assert "results" in data
    assert len(data["results"]) > 0
    assert "status" in data


def test_search_sensitivity():
    r = client.post("/api/search", json={
        "search": {"mode": "sensitivity", "sweep_param": "group_size", "sweep_values": [4, 8]},
    })
    assert r.status_code == 200
    assert len(r.json()["results"]) == 2


def test_static_files():
    assert client.get("/static/styles.css").status_code == 200
    assert client.get("/static/app.js").status_code == 200
```

- [ ] **Step 2: Run all tests**

```bash
pytest tests/ -x -v
```

Expected: All tests pass.

- [ ] **Step 3: Run linter**

```bash
ruff check src/rl_perf/ui/ && ruff format --check src/rl_perf/ui/
```

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "test: add FastAPI endpoint tests for new UI backend"
```

---

## Summary

| Task | Description | Key Output |
|------|-------------|------------|
| 1 | Delete Gradio, update deps | Clean slate |
| 2 | FastAPI backend (api.py) | 5 working REST endpoints |
| 3 | HTML structure (index.html) | Complete page with all form fields |
| 4 | CSS design system (styles.css) | Tokens, layout, components, responsive |
| 5 | JavaScript (app.js) | Accordion, tabs, fetch, Plotly charts |
| 6 | Visual polish + audit | Impeccable-compliant design |
| 7 | Tests + verification | API tests, all passing |

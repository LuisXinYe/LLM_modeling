# Baseline UI Audit Report

**Date:** 2026-03-30
**Auditor:** Claude (automated)
**Scope:** `src/rl_perf/ui/` -- all 10 GUI source files
**Design context:** `.impeccable.md` (Huawei professional dashboard, warm-neutral, data-first)

---

## Overall Score: 5 / 20

| Dimension | Score (0-4) | Summary |
|-----------|-------------|---------|
| Accessibility | 1 | No ARIA, no focus states, no semantic structure |
| Performance | 2 | Lightweight Python-side; Plotly bundle is heavy but unavoidable |
| Theming | 0 | No theme applied, no design tokens, no custom CSS |
| Responsive Design | 1 | Gradio default fluid layout only; no breakpoints, no touch sizing |
| Anti-Patterns | 1 | Multiple AI-slop tells; default Gradio look throughout |

---

## 1. Accessibility (1/4)

### Positive Findings
- Gradio components have `label=` on most inputs (provides basic screen reader support)
- Conditional visibility (`gr.update(visible=...)`) prevents irrelevant fields from confusing users
- Plotly figures include hover text with meaningful data

### Issues

| Sev | Issue | Location | Recommendation |
|-----|-------|----------|----------------|
| P0 | **No ARIA landmarks or roles** -- entire app is flat `div` soup. No `role="main"`, no `role="navigation"` for tabs, no `aria-live` for results area | `app.py:138-168` | Add `elem_id` to major sections; inject ARIA via custom CSS/JS |
| P0 | **KPI cards are raw Markdown** -- screen readers see `### Epoch Time` as heading level 3 with no semantic grouping. Four KPIs render as consecutive headings with no container role | `results.py:23-33`, `app.py:373-393` | Replace with `gr.HTML` using `<dl>`/`<dd>` or `role="status"` |
| P1 | **No keyboard navigation testing** -- Gradio tabs use default browser behavior; custom topology plot (Plotly scatter) is not keyboard-navigable | `topology.py:125-237` | Add `tabindex` and keyboard event handlers to interactive elements |
| P1 | **No focus indicators on custom elements** -- "Run Prediction" and "Run Search" buttons rely on browser defaults which vary | `app.py:152`, `tab_search.py:82` | Custom CSS `:focus-visible` ring |
| P1 | **Color-only encoding in Pareto plot** -- OOM (red x), feasible (gray), Pareto (green) use color alone. Red-green colorblind users cannot distinguish OOM from Pareto | `plots.py:234-276` | Already uses different marker symbols (x vs circle) for OOM -- good. But feasible vs Pareto is circle vs circle, distinguished only by color and size |
| P2 | **Sensitivity bar colors have no secondary encoding** -- green/amber/red bars with no pattern or label indicating status | `plots.py:342-350` | Add text annotation or hatching pattern for infeasible bars |
| P2 | **Contrast ratios untested** -- gray text on white background (`_GRAY = "#9ca3af"`) used for feasible non-Pareto points likely fails WCAG AA (4.5:1 for normal text) | `plots.py:24` | Verify contrast; darken to `#6b7280` minimum |
| P3 | **Missing `alt` text for topology plot** -- Plotly figures have no accessible description | `tab_hardware.py:163` | Provide `label=` with meaningful description |

---

## 2. Performance (2/4)

### Positive Findings
- All computation is Python-side (no client-side heavy processing)
- Plotly figures are generated on demand, not pre-rendered
- Conditional imports (`import plotly.graph_objects as _go` inside `_run_search`) avoid unnecessary loading

### Issues

| Sev | Issue | Location | Recommendation |
|-----|-------|----------|----------------|
| P1 | **Redundant topology recomputation** -- topology figure regenerates on every change to TP, PP, DP, EP, total_devices, and hw_dropdown (6 change listeners). Rapid slider/input changes cause 6 redundant calls | `tab_hardware.py:199-204` | Add debounce or a single "Update Topology" button |
| P2 | **Full Plotly bundle loaded** -- Plotly.js is ~3.5 MB minified. For 4 chart types (bar, scatter, stacked bar, scatter+text) this is heavy | `plots.py`, `topology.py`, `tab_search.py` | Consider `plotly.graph_objects` with `include_plotlyjs='cdn'` or Plotly partial bundles if Gradio supports it |
| P2 | **Empty figures created on load** -- 4 empty `go.Figure()` instances created at tab render time, each triggering a Plotly render cycle for blank charts | `results.py:9-12`, `tab_search.py:22-25`, `tab_hardware.py:25-32` | Use `None` initial value or lazy-load pattern |
| P3 | **DP auto-compute fires 4 separate listeners** -- changing `total_devices` triggers `_update_dp` + `_update_nodes` + `_update_topology`, each as separate round-trips | `tab_hardware.py:79-88, 133-138, 199-204` | Consolidate into a single callback |
| P3 | **Config directory resolved at import time** -- `_CONFIGS_DIR` computed via `Path(__file__).resolve().parent...` in 3 files independently | `app.py:27`, `tab_model.py:12`, `tab_hardware.py:13` | Centralize in a shared constants module |

---

## 3. Theming (0/4)

### Positive Findings
- `plots.py` defines a color palette (`_PURPLE`, `_GREEN`, `_RED`, etc.) that partially aligns with `.impeccable.md`
- Plotly figures use `template="plotly_white"` consistently

### Issues

| Sev | Issue | Location | Recommendation |
|-----|-------|----------|----------------|
| P0 | **No Gradio theme applied in `create_app()`** -- `gr.Blocks()` has no `theme=` argument. Theme is only set in `launch()` via `app.launch(theme=gr.themes.Soft())` which is too late for `create_app()` callers | `app.py:138` vs `app.py:688` | Move `theme=gr.themes.Soft()` into `gr.Blocks(theme=...)` |
| P0 | **No custom CSS at all** -- zero `css=` parameter, zero `elem_classes`, zero `elem_id`. The entire UI uses stock Gradio Soft theme defaults | `app.py:138-140` | Add `css=` parameter with design tokens from `.impeccable.md` |
| P0 | **Huawei red (`#CF0A2C`) never appears** -- the brand accent color specified in `.impeccable.md` is completely absent from the codebase | All files | Apply to primary buttons, active tab indicators, KPI highlights |
| P0 | **No design token system** -- colors are hardcoded strings in `plots.py` and `topology.py`. No CSS custom properties, no shared constants for UI chrome | `plots.py:20-28`, `topology.py:21-38` | Create `_tokens.py` or CSS variables for all design tokens |
| P1 | **KPI cards have no visual styling** -- rendered as plain Markdown (`### Epoch Time\n\n**value**`). No background, no border, no padding, no visual hierarchy | `app.py:373-393`, `results.py:23-33` | Implement as styled `gr.HTML` with background cards, large number typography |
| P1 | **No dark mode consideration** -- `.impeccable.md` says "Light mode" but no explicit light theme enforcement. Gradio may follow OS preference | `app.py:138` | Explicitly set light mode; add `@media (prefers-color-scheme: dark)` override if needed |
| P1 | **Title is plain Markdown** -- `gr.Markdown("# rl-perf\nRL Training Performance Modeling")` with no brand styling, no logo, no visual weight | `app.py:141` | Style header with brand color, proper typography, possibly a small logo |
| P2 | **Plotly chart backgrounds are pure white** -- `.impeccable.md` specifies "warm neutral backgrounds (not pure white)" | `plots.py` (all figures) | Set `plot_bgcolor` and `paper_bgcolor` to warm off-white (e.g., `#faf8f5`) |
| P2 | **No consistent spacing rhythm** -- `gr.Group()` and `gr.Row()` use Gradio defaults with no `elem_classes` for custom spacing | All tab files | Define spacing scale in CSS (4px, 8px, 16px, 24px, 32px) |

---

## 4. Responsive Design (1/4)

### Positive Findings
- Gradio's `gr.Row()` and `gr.Column(scale=)` provide basic fluid layout
- 7:3 main/sidebar split is reasonable for desktop

### Issues

| Sev | Issue | Location | Recommendation |
|-----|-------|----------|----------------|
| P0 | **Fixed 7:3 column split breaks on mobile** -- sidebar and main content will be crushed on narrow screens. No breakpoint to stack vertically | `app.py:143-145, 158` | Add CSS media query to stack columns below 768px |
| P1 | **Number inputs have no min-width** -- small precision=0 number fields (TP, PP, DP, EP, CP) render very narrow on mobile, making them hard to tap | `tab_hardware.py:93-108` | Set minimum touch target size (44x44px per WCAG 2.5.5) |
| P1 | **Plotly charts have fixed margins** -- `margin=dict(l=80, r=40, t=60, b=40)` hardcoded in pixels. On small screens, margins consume most of the plot area | `plots.py:103, 188, 296, 381` | Use relative margins or responsive Plotly config |
| P2 | **Chatbot height hardcoded** -- `gr.Chatbot(label="Chat", height=400)` is 400px fixed. On short screens this pushes content below fold | `app.py:160` | Use `height="auto"` or CSS `max-height` with overflow |
| P2 | **Topology plot marker size fixed at 30px** -- `marker=dict(symbol="square", size=30)` does not scale for small screens or large meshes | `topology.py:210-215` | Scale marker size based on total ranks |
| P3 | **Dataframe table not scrollable** -- `gr.Dataframe` with 10 columns may overflow horizontally on narrow screens | `tab_search.py:97-113` | Wrap in scrollable container or reduce visible columns on mobile |

---

## 5. Anti-Patterns (1/4)

### Positive Findings
- Conditional field visibility (MLA/MoE/mHC rows) is a good progressive disclosure pattern
- Auto-computed fields (DP, Nodes, Total Responses) reduce user error
- Color palette in plots is semantically meaningful (green=feasible, red=OOM)

### Issues

| Sev | Issue | Location | Recommendation |
|-----|-------|----------|----------------|
| P0 | **"AI Assistant coming soon" placeholder** -- empty chatbot + stub text is the definition of AI slop. Shipping an empty chat widget signals "this was generated, not designed" | `app.py:159-167` | Remove AI sidebar entirely until functional, or replace with a static "Analysis Notes" panel |
| P0 | **Default Gradio aesthetic** -- the entire UI looks like a Gradio tutorial example. No custom styling, no brand identity, no visual hierarchy beyond Markdown headings | Entire codebase | Comprehensive CSS overhaul per `.impeccable.md` |
| P1 | **Generic emoji in AI sidebar stub** -- emojis in `ai_sidebar.py` and `chat_respond()` | `ai_sidebar.py:12,19,27,57-63` | Remove decorative emojis; use icons or nothing |
| P1 | **"Run Prediction" is the only CTA** -- no visual hierarchy between primary action (Run Prediction) and secondary actions (Load Model, Run Search). All use `variant="primary"` | `app.py:152`, `tab_model.py:58`, `tab_search.py:82` | Differentiate: primary (Run Prediction), secondary (Load Model), tertiary (Run Search) |
| P1 | **Results hidden until first run** -- `visible=False` on `results_container` means 60% of the page is blank on first load. No placeholder, no example, no onboarding | `results.py:19` | Show example results or a placeholder illustration |
| P2 | **Section headers are just `gr.Markdown("### ...")`** -- no visual treatment (background, border-bottom, icon). Reads as a wall of form fields | All tab files | Style section headers with subtle background or left border accent |
| P2 | **Error messages displayed as Markdown** -- `f"**Error:** {e}"` renders as bold text indistinguishable from normal content. No error styling, no color, no icon | `app.py:418`, `tab_model.py:255` | Use `gr.Warning()` or styled error HTML with red accent |
| P2 | **"Topology Preview" always visible** -- even when it shows an empty plot with default axes. Should be hidden or collapsed until parallelism is configured | `tab_hardware.py:161-166` | Collapse by default; show after first valid config |
| P3 | **Validation message is just Markdown** -- `validation_msg = gr.Markdown("")` for warnings like "TP*PP*DP*EP != total" has no visual warning treatment | `tab_hardware.py:165-166` | Style as a warning banner with amber background |

---

## Summary of P0 Issues (Must Fix)

1. **No ARIA landmarks** -- accessibility violation (`app.py`)
2. **KPI cards are unstyled Markdown** -- no visual hierarchy for the most important data (`results.py`, `app.py`)
3. **Theme not applied in `create_app()`** -- `gr.Blocks()` missing `theme=` (`app.py:138`)
4. **Zero custom CSS** -- no design tokens, no brand identity (`app.py`)
5. **Huawei red accent absent** -- brand color from spec never used (all files)
6. **No design token system** -- colors hardcoded as strings (`plots.py`, `topology.py`)
7. **Layout breaks on mobile** -- fixed column split with no responsive breakpoints (`app.py`)
8. **AI sidebar placeholder is shipped** -- empty chatbot widget with "coming soon" text (`app.py:159-167`)
9. **Default Gradio aesthetic** -- looks like a tutorial, not a professional tool (entire codebase)

---

## Recommended Fix Order

1. **Theming first** (P0s 3-6): Apply theme to `gr.Blocks`, create design tokens, add custom CSS with Huawei red
2. **Layout** (P0s 7, 9): Responsive breakpoints, KPI card styling, section hierarchy
3. **Content cleanup** (P0 8): Remove or replace AI sidebar placeholder
4. **Accessibility** (P0 1): ARIA landmarks, focus states, semantic HTML for KPI cards
5. **Polish**: Chart backgrounds, spacing rhythm, error styling, progressive disclosure improvements

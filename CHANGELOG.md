# Changelog

## [Unreleased]

### Fixed
- When `ffn_linear` is unset, backward FFN `compute_class` now resolves from the
  `gradients` role (was `activations`); benign when gradients == activations (all default paths)
- `FEASIBLE` status now reflects both time budget AND memory feasibility
- Builder rejects invalid TP/PP configurations instead of silently truncating
- Simulator now models compute–communication overlap and per-fabric bandwidth
  contention (same-fabric collectives time-share; NVLink vs NIC independent);
  the stale "no overlapping communication" limitation is removed

### Added
- **Low-precision (FP8/FP4) training modeling.** Per-tensor-role `PrecisionConfig`
  (weights/activations/gradients/comm) with fine-grained block scaling and
  stochastic Hadamard; quantize/dequant/Hadamard overhead ops; per-precision peak
  TFLOPS in `roofline_time(compute_class=...)`; precision-aware GEMM memory;
  low-precision gradient comm; error-feedback buffers; mixed/periodic
  high-precision steps
- `model.compare_precision()` — side-by-side recipe comparison (step time, speedup,
  comm reduction, exposed comm by fabric, peak memory, feasibility)
- `SimResult.exposed_comm_by_fabric` — per-fabric communication not hidden under compute
- `examples/demo_low_precision.py` — low-precision recipe comparison demo
- `--format json` flag for `targets` and `check` CLI commands
- `feasible` field on `TargetReport` (composite of time + memory)
- `feasibility_check` now accepts optional `time_budget_hours`
- CLI error handling: friendly messages instead of raw tracebacks
- ParallelismConfig validates tp/pp/dp/ep/cp >= 1
- Builder validates TP divides num_heads/num_kv_heads/intermediate_size
- Docstrings for all public classes and functions
- `examples/demo.py` — exploration workflow demo
- `docs/architecture.md` — architecture deep-dive
- `docs/config-reference.md` — configuration field reference
- `docs/result-interpretation.md` — report interpretation guide
- `docs/calibration-guide.md` — calibration coefficient guide
- `docs/troubleshooting.md` — common errors and fixes

### Changed
- `format_table` shows `NOT FEASIBLE: OOM`, `NOT FEASIBLE: OVER TIME`, or combined
- `check` command help text now shows hardware shortnames

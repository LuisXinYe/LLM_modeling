# RL Performance Modeling (rl-perf)

## Project Overview
RL 训练性能建模工具。给定模型、卡数、数据量，推导 epoch 时间上限和各团队 TPS 指标。

## Tech Stack
- Python 3.10+ (use venv: `source .venv/bin/activate`)
- No ML framework dependencies (pure Python + numpy)
- CLI: typer, Config: pydantic + YAML

## Key Architecture
- L1 (ops.py): 模块化算子 cost model, roofline-based, 可被 benchmark 校准
- L2 (builder.py + simulator.py): Template DAG + 多流拓扑模拟
- L3 (pipeline.py): 两阶段 pipeline (generation → queue → training)

## Development
- Activate venv: `source .venv/bin/activate`
- Tests: `pytest tests/ -v`
- Lint: `ruff check src/ && ruff format src/`
- Install: `pip install -e ".[dev]"`

## Conventions
- All FLOPs formulas must have docstring citing source
- Hardware constants in configs/hardware/*.yaml, NOT hardcoded
- One op function per attention/FFN variant in ops.py

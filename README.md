# rl-perf: RL Training Performance Modeling

Given model + hardware + RL config, predict epoch time and derive TPS targets for inference and training teams.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

```bash
rl-perf targets --model configs/models/llama3_1_8b.yaml \
                --hardware configs/hardware/ascend_910c.yaml \
                --devices 64 --prompts 10000 --group-size 8 --time-budget 24
```

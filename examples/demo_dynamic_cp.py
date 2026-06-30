#!/usr/bin/env python3
"""Dynamic context parallelism (dynamic-CP) demo.

Compares two end-to-end recipes over a variable-length distribution:

    * Static CP  + packing + PP bubble  (every sequence sharded by max_cp)
    * Dynamic CP + packing + PP bubble  (per-bucket CP by sequence length)

using the simplified analytical model in llm_perf.dynamic_cp. Reports the
amortized step time, packing efficiency, PP-bubble share, achieved MFU /
TFLOPS-per-GPU and per-rank peak memory for each recipe.

Usage:
    python examples/demo_dynamic_cp.py
    python examples/demo_dynamic_cp.py --model deepseekv3_671b --cp 8 --pp 4 \
        --avg 4096 --std 8192 --max-len 65536 --total-ranks 256
"""

import argparse
from pathlib import Path

from llm_perf.config import (
    ParallelismConfig,
    WorkloadConfig,
    load_hardware_config,
    load_model_config,
)
from llm_perf.dynamic_cp import compare_cp_strategies, lognormal_buckets

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "configs"


def main():
    p = argparse.ArgumentParser(description="Dynamic-CP static-vs-dynamic demo")
    p.add_argument("--model", default="llama3_1_8b")
    p.add_argument("--hardware", default="ascend_910c")
    p.add_argument("--tp", type=int, default=2)
    p.add_argument("--cp", type=int, default=8, help="max CP degree available")
    p.add_argument("--pp", type=int, default=8, help="pipeline depth")
    p.add_argument("--vstages", type=int, default=1, help="virtual stages V")
    p.add_argument("--avg", type=float, default=4096, help="mean sequence length")
    p.add_argument("--std", type=float, default=8192, help="sequence length std")
    p.add_argument("--max-len", type=float, default=65536, help="max sequence length")
    p.add_argument("--buckets", type=int, default=8)
    p.add_argument("--total-ranks", type=int, default=8,
                   help="CP/DP rank pool the work is amortized over")
    p.add_argument("--token-budget", type=float, default=None,
                   help="packing budget in tokens (default: max-len)")
    p.add_argument("--global-batch-seqs", type=int, default=64,
                   help="total sequences in global batch")
    args = p.parse_args()

    mc = load_model_config(str(CONFIGS / "models" / f"{args.model}.yaml"))
    hw = load_hardware_config(str(CONFIGS / "hardware" / f"{args.hardware}.yaml"))
    par = ParallelismConfig(tp=args.tp, cp=args.cp, dp=1, pp=args.pp)
    wl = WorkloadConfig(group_size=1)

    buckets = lognormal_buckets(args.avg, args.std, args.max_len, args.buckets)
    r = compare_cp_strategies(
        mc, hw, par, wl, buckets,
        total_ranks=args.total_ranks,
        pp=args.pp,
        v=args.vstages,
        token_budget=args.token_budget,
        global_batch_seqs=args.global_batch_seqs,
    )

    print(f"Model: {mc.name} | tp={args.tp} pp={args.pp} V={args.vstages} "
          f"| max CP={r['max_cp']} | pool R={r['total_ranks']}")
    print("=" * 80)

    def show(name, s):
        print(f"  {name:<8} step={s['step_s']*1000:>8.2f}ms  m={s['m']:>3}  "
              f"bubble={s['bubble_ratio']*100:>4.1f}%  MFU={s['mfu']*100:>4.1f}%  "
              f"TFLOPS/GPU={s['tflops_per_gpu']:>6.1f}  peakmem={s['peak_mem_gb']:>5.1f}GB "
              f"{'OK' if s['feasible'] else 'OOM'}  imbal={s['imbalance']:.2f}")

    show("Static", r["static"])
    show("Dynamic", r["dynamic"])
    print("-" * 80)
    print(f"  Speedup (static/dynamic step): {r['speedup']:.2f}x   "
          f"TFLOPS/GPU ratio: {r['tflops_ratio']:.2f}x")
    st, dy = r["static"], r["dynamic"]
    print(f"  CP-group init (one-time, pre-built at init): "
          f"static groups={st['cp_groups']} {st['cp_init_s']*1000:.0f}ms/"
          f"{st['cp_init_mem_gb']*1000:.0f}MB  |  "
          f"dynamic groups={dy['cp_groups']} {dy['cp_init_s']*1000:.0f}ms/"
          f"{dy['cp_init_mem_gb']*1000:.0f}MB  |  "
          f"dynamic extra: +{r['cp_init_extra_s']*1000:.0f}ms / "
          f"+{r['cp_init_extra_mem_gb']*1000:.0f}MB (amortized over all steps)")
    print("  Note: the headline speedup is sensitive to --global-batch-seqs "
          "(observed to drift ~1.5-1.9x over a swept range) — treat the single "
          "number above as illustrative, not a guaranteed constant.")


if __name__ == "__main__":
    main()

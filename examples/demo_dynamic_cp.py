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
    p.add_argument("--tp", type=int, default=8)
    p.add_argument("--cp", type=int, default=8, help="max CP degree available")
    p.add_argument("--pp", type=int, default=1, help="pipeline depth (for PP bubble)")
    p.add_argument("--avg", type=float, default=4096, help="mean sequence length")
    p.add_argument("--std", type=float, default=8192, help="sequence length std")
    p.add_argument("--max-len", type=float, default=65536, help="max sequence length")
    p.add_argument("--buckets", type=int, default=8)
    p.add_argument("--total-ranks", type=int, default=64,
                   help="CP/DP rank pool the work is amortized over")
    p.add_argument("--token-budget", type=float, default=None,
                   help="packing budget in tokens (default: max-len)")
    p.add_argument("--micro-batches", type=int, default=None,
                   help="micro-batches per pipeline fill (default: max(pp, grad-accum))")
    args = p.parse_args()

    mc = load_model_config(str(CONFIGS / "models" / f"{args.model}.yaml"))
    hw = load_hardware_config(str(CONFIGS / "hardware" / f"{args.hardware}.yaml"))
    par = ParallelismConfig(tp=args.tp, cp=args.cp, dp=1, pp=args.pp)
    wl = WorkloadConfig(group_size=1)

    buckets = lognormal_buckets(args.avg, args.std, args.max_len, args.buckets)
    r = compare_cp_strategies(
        mc, hw, par, wl, buckets,
        total_ranks=args.total_ranks,
        token_budget=args.token_budget,
        num_micro_batches=args.micro_batches,
    )

    print(f"Model: {mc.name} | TP={args.tp} | max CP={r['max_cp']} | PP={args.pp}")
    print(f"quota={r['quota']:.0f} tok | packing budget={r['token_budget']:.0f} tok "
          f"| micro-batches={r['num_micro_batches']} | rank pool={r['total_ranks']}")
    print(f"Length dist: avg={args.avg:.0f} std={args.std:.0f} max={args.max_len:.0f}")
    print("=" * 78)
    print(f"{'seq_len':>9} {'frac':>7} {'cp(dyn)':>8} {'cp(stat)':>9} "
          f"{'cost_dyn':>10} {'cost_stat':>10}  (rank-ms)")
    print("-" * 78)
    for bd, bs in zip(r["dynamic"]["buckets"], r["static"]["buckets"]):
        print(f"{bd['seq_len']:>9.0f} {bd['fraction']:>7.3f} {bd['cp']:>8} "
              f"{bs['cp']:>9} {bd['cost_rank_s']*1000:>10.1f} "
              f"{bs['cost_rank_s']*1000:>10.1f}")
    print("=" * 78)

    def show(name, s):
        print(f"  {name:<8} step={s['step_s']*1000:>8.3f} ms  "
              f"η_pack={s['packing_eff']:.2f}  bubble={s['bubble_ratio']*100:>4.1f}%  "
              f"MFU={s['mfu']*100:>4.1f}%  TFLOPS/GPU={s['tflops_per_gpu']:>6.1f}  "
              f"peakmem={s['peak_mem_gb']:>5.1f}GB {'OK' if s['feasible'] else 'OOM'}")

    show("Static", r["static"])
    show("Dynamic", r["dynamic"])
    print("-" * 78)
    print(f"  Speedup (static step / dynamic step): {r['speedup']:.2f}x")
    print(f"  TFLOPS/GPU ratio (dynamic / static):  {r['tflops_ratio']:.2f}x")
    print()
    print("Notes:")
    print("  - rank-seconds = cp * per-rank time (resource occupancy); the static-vs-")
    print("    dynamic gap comes from the CP assignment. Packing (η) and the PP bubble")
    print("    are overheads layered on BOTH recipes — they shift absolute step time")
    print("    and MFU, not the relative gain.")
    print("  - peak memory is per-rank (O(S) axis): static over-shards short sequences")
    print("    (lower mem, wasted ranks); dynamic keeps cp small for them (higher mem).")


if __name__ == "__main__":
    main()

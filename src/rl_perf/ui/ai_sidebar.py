"""AI chat sidebar for rl-perf GUI.

Provides result summary and chat stubs. Full LLM integration to be added later.
"""

from rl_perf.report import TargetReport


def format_result_summary(report: TargetReport) -> str:
    """Generate human-readable summary of prediction results."""
    lines = []
    lines.append(
        f"📌 **Bottleneck: {report.bottleneck}** (slack {report.bottleneck_slack:.0%})"
    )

    if report.bottleneck == "generation":
        lines.append("Generation dominates — decode phase is memory-bandwidth bound.")
        lines.append("")
        lines.append("💡 **Suggestions:**")
        lines.append("1. Enable speculative decoding (if MTP supported)")
        lines.append("2. Increase generation DP to parallelize across more devices")
        lines.append("3. Reduce avg_response_len if possible")
    else:
        lines.append("Training dominates — consider increasing training parallelism.")
        lines.append("")
        lines.append("💡 **Suggestions:**")
        lines.append("1. Increase DP or PP for training")
        lines.append("2. Enable gradient accumulation to increase effective batch size")
        lines.append("3. Consider activation checkpointing to reduce memory pressure")

    mem = report.memory
    if not mem.train_feasible:
        lines.append("")
        lines.append(
            f"⚠️ **Training OOM**: {mem.total_train_gb:.1f} GB > {mem.usable_hbm_gb:.1f} GB"
        )
        lines.append(f"   Optimizer: {mem.optimizer_gb:.1f} GB — try ZeRO stage 2+")
        if mem.ref_model_gb > 0:
            lines.append(f"   Ref model: {mem.ref_model_gb:.1f} GB — try CPU offload")
    if not mem.gen_feasible:
        lines.append("")
        lines.append(
            f"⚠️ **Generation OOM**: {mem.total_gen_gb:.1f} GB > {mem.usable_hbm_gb:.1f} GB"
        )
        lines.append(
            f"   KV cache: {mem.kv_cache_gb:.1f} GB — reduce gen_batch_size or max_response_len"
        )

    return "\n".join(lines)


def chat_respond(message: str, history: list) -> str:
    """Process a chat message. Currently returns static help."""
    msg = message.lower().strip()

    if any(w in msg for w in ["help", "帮助", "怎么"]):
        return (
            "I can help with:\n\n"
            "📝 **Describe your setup**: e.g., '128 cards, Qwen 72B, 100K prompts, 24h budget'\n"
            "🔍 **Ask for optimization**: e.g., 'how to reduce generation time?'\n"
            "📊 **Request analysis**: e.g., 'compare colocated vs separated'\n\n"
            "*(Full AI assistant requires LLM API configuration — coming soon)*"
        )

    return (
        "Full AI assistant with natural language understanding is coming soon.\n\n"
        "For now, use the configuration tabs and click **Run Prediction** to see results."
    )

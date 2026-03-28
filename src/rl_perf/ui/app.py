"""Main Gradio application for rl-perf GUI."""

from __future__ import annotations

from pathlib import Path

import gradio as gr

from rl_perf.config import (
    HardwareConfig,
    LayerConfig,
    ModelConfig,
    ParallelismConfig,
    RLConfig,
    load_hardware_config,
)
from rl_perf.model import RLPerformanceModel
from rl_perf.search import pareto_search, sensitivity_sweep
from rl_perf.ui.plots import (
    build_memory_figure,
    build_pareto_figure,
    build_sensitivity_figure,
    build_timeline_figure,
)
from rl_perf.ui import tab_model, tab_hardware, tab_rl, tab_search, results

_CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "configs"

_HW_TEMPLATES: dict[str, str] = {
    "Ascend 910C": "ascend_910c",
    "CloudMatrix 384": "cloudmatrix_384",
}


def _build_hw_config(hw_name: str) -> HardwareConfig:
    """Load a HardwareConfig from the named profile."""
    stem = _HW_TEMPLATES.get(hw_name)
    if not stem:
        raise ValueError(f"Unknown hardware profile: {hw_name}")
    return load_hardware_config(str(_CONFIGS_DIR / "hardware" / f"{stem}.yaml"))


def _build_model_config(
    name,
    hidden_size,
    vocab_size,
    num_layers,
    dtype,
    attention,
    ffn,
    residual,
    num_heads,
    num_kv_heads,
    head_dim,
    intermediate_size,
    kv_compression_dim,
    query_compression_dim,
    rope_dim,
    window_size,
    num_experts,
    top_k,
    num_shared_experts,
    expert_intermediate_size,
    shared_intermediate_size,
    mhc_expansion,
) -> ModelConfig:
    """Build a ModelConfig from individual GUI field values."""
    layer = LayerConfig(
        attention=str(attention),
        num_heads=int(num_heads),
        num_kv_heads=int(num_kv_heads),
        head_dim=int(head_dim),
        ffn=str(ffn),
        intermediate_size=int(intermediate_size),
        num_experts=int(num_experts),
        num_shared_experts=int(num_shared_experts),
        top_k=int(top_k),
        expert_intermediate_size=int(expert_intermediate_size),
        shared_intermediate_size=int(shared_intermediate_size),
        kv_compression_dim=int(kv_compression_dim),
        query_compression_dim=int(query_compression_dim),
        rope_dim=int(rope_dim),
        window_size=int(window_size),
        residual=str(residual),
        mhc_expansion=int(mhc_expansion),
    )
    return ModelConfig(
        name=str(name),
        hidden_size=int(hidden_size),
        vocab_size=int(vocab_size),
        num_layers=int(num_layers),
        dtype=str(dtype),
        default_layer=layer,
    )


def _build_rl_config(
    total_prompts,
    group_size,
    avg_prompt_len,
    avg_response_len,
    max_response_len,
    std_response_len,
    train_micro_batch_size,
    grad_accumulation_steps,
    gen_batch_size,
    colocated,
    reference_model,
    ref_offload_cpu,
    speculative_decoding,
    mtp_acceptance_len,
) -> RLConfig:
    """Build an RLConfig from individual GUI field values."""
    std_val = int(std_response_len) if std_response_len else None
    if std_val == 0:
        std_val = None
    mtp_val = int(mtp_acceptance_len) if speculative_decoding else None
    return RLConfig(
        total_prompts=int(total_prompts),
        group_size=int(group_size),
        avg_prompt_len=int(avg_prompt_len),
        avg_response_len=int(avg_response_len),
        max_response_len=int(max_response_len),
        std_response_len=std_val,
        train_micro_batch_size=int(train_micro_batch_size),
        gradient_accumulation_steps=int(grad_accumulation_steps),
        gen_batch_size=int(gen_batch_size),
        colocated=(colocated == "Colocated"),
        reference_model=bool(reference_model),
        ref_offload_cpu=bool(ref_offload_cpu),
        use_speculative_decoding=bool(speculative_decoding),
        mtp_acceptance_len=mtp_val,
    )


def create_app() -> gr.Blocks:
    """Build and return the Gradio Blocks application."""
    with gr.Blocks(
        title="rl-perf -- RL Training Performance Modeling",
    ) as app:
        gr.Markdown("# rl-perf\nRL Training Performance Modeling")

        with gr.Row():
            # ---- Left column: main content ----
            with gr.Column(scale=7):
                with gr.Tabs():
                    mc = tab_model.build_tab()
                    hc = tab_hardware.build_tab()
                    rc = tab_rl.build_tab()
                    sc = tab_search.build_tab()

                run_btn = gr.Button("Run Prediction", variant="primary", size="lg")
                prediction_status = gr.Markdown("")

                res = results.build_results()

            # ---- Right column: AI sidebar placeholder ----
            with gr.Column(scale=3):
                gr.Markdown("### AI Assistant")
                gr.Chatbot(label="Chat", height=400)
                gr.Textbox(
                    placeholder="Ask about your configuration...",
                    label="Message",
                )
                gr.Markdown(
                    "*AI assistant coming soon.*",
                )

        # ======================================================
        # Wire: Run Prediction
        # ======================================================

        # Collect all model field inputs
        _model_inputs = [
            mc["name"],
            mc["hidden_size"],
            mc["vocab_size"],
            mc["num_layers"],
            mc["dtype"],
            mc["attention"],
            mc["ffn"],
            mc["residual"],
            mc["num_heads"],
            mc["num_kv_heads"],
            mc["head_dim"],
            mc["intermediate_size"],
            mc["kv_compression_dim"],
            mc["query_compression_dim"],
            mc["rope_dim"],
            mc["window_size"],
            mc["num_experts"],
            mc["top_k"],
            mc["num_shared_experts"],
            mc["expert_intermediate_size"],
            mc["shared_intermediate_size"],
            mc["mhc_expansion"],
        ]

        _hw_inputs = [
            hc["hw_dropdown"],
            hc["total_devices"],
            hc["tp"],
            hc["pp"],
            hc["dp"],
            hc["ep"],
            hc["cp"],
            hc["cp_type"],
            hc["sp"],
            hc["zero_stage"],
            hc["pp_schedule"],
            hc["recompute_attention"],
            hc["full_recomputation"],
            hc["optimizer_offload"],
            hc["activation_offload"],
        ]

        _rl_inputs = [
            rc["total_prompts"],
            rc["group_size"],
            rc["avg_prompt_len"],
            rc["avg_response_len"],
            rc["max_response_len"],
            rc["std_response_len"],
            rc["train_micro_batch_size"],
            rc["grad_accumulation_steps"],
            rc["gen_batch_size"],
            rc["colocated"],
            rc["reference_model"],
            rc["ref_offload_cpu"],
            rc["speculative_decoding"],
            rc["mtp_acceptance_len"],
        ]

        _all_prediction_inputs = _model_inputs + _hw_inputs + _rl_inputs

        def _run_prediction(
            # Model fields (22)
            m_name,
            m_hidden,
            m_vocab,
            m_layers,
            m_dtype,
            m_attn,
            m_ffn,
            m_residual,
            m_nheads,
            m_nkv,
            m_hdim,
            m_inter,
            m_kv_comp,
            m_q_comp,
            m_rope,
            m_window,
            m_nexp,
            m_topk,
            m_nshared,
            m_exp_inter,
            m_shared_inter,
            m_mhc,
            # Hardware fields (15)
            hw_name,
            total_dev,
            tp,
            pp,
            dp,
            ep,
            cp,
            cp_type,
            sp,
            zero_stage,
            pp_sched,
            recomp_attn,
            full_recomp,
            opt_offload,
            act_offload,
            # RL fields (14)
            total_prompts,
            group_size,
            avg_plen,
            avg_rlen,
            max_rlen,
            std_rlen,
            train_mbs,
            grad_acc,
            gen_bs,
            colocated,
            ref_model,
            ref_offload,
            spec_dec,
            mtp_len,
        ):
            try:
                model_cfg = _build_model_config(
                    m_name,
                    m_hidden,
                    m_vocab,
                    m_layers,
                    m_dtype,
                    m_attn,
                    m_ffn,
                    m_residual,
                    m_nheads,
                    m_nkv,
                    m_hdim,
                    m_inter,
                    m_kv_comp,
                    m_q_comp,
                    m_rope,
                    m_window,
                    m_nexp,
                    m_topk,
                    m_nshared,
                    m_exp_inter,
                    m_shared_inter,
                    m_mhc,
                )
                hw_cfg = _build_hw_config(hw_name)

                tp_v, pp_v, dp_v, ep_v, cp_v = (
                    int(tp),
                    int(pp),
                    int(dp),
                    int(ep),
                    int(cp),
                )
                total_dev_v = int(total_dev)

                train_parallel = ParallelismConfig(
                    tp=tp_v,
                    pp=pp_v,
                    dp=dp_v,
                    ep=ep_v,
                    cp=cp_v,
                    cp_type=str(cp_type),
                    sp=bool(sp),
                    zero_stage=int(zero_stage),
                    pp_schedule=str(pp_sched),
                    recompute_attention=bool(recomp_attn),
                    full_recomputation=bool(full_recomp),
                    optimizer_offload=bool(opt_offload),
                    activation_offload=bool(act_offload),
                )
                gen_dp = total_dev_v // tp_v if tp_v > 0 else 1
                gen_parallel = ParallelismConfig(tp=tp_v, pp=1, dp=gen_dp)

                rl_cfg = _build_rl_config(
                    total_prompts,
                    group_size,
                    avg_plen,
                    avg_rlen,
                    max_rlen,
                    std_rlen,
                    train_mbs,
                    grad_acc,
                    gen_bs,
                    colocated,
                    ref_model,
                    ref_offload,
                    spec_dec,
                    mtp_len,
                )

                perf_model = RLPerformanceModel(model_cfg, hw_cfg)
                report = perf_model.derive_targets(
                    total_dev_v,
                    rl_cfg,
                    gen_parallel,
                    train_parallel,
                    time_budget_hours=None,
                )

                # Format KPI cards
                feasible_str = "FEASIBLE" if report.feasible else "NOT FEASIBLE"
                kpi_epoch = (
                    f"### Epoch Time\n\n"
                    f"**{report.epoch_time_hours:.2f}** hours\n\n"
                    f"*{feasible_str}*"
                )
                kpi_gen = (
                    f"### Gen TPS\n\n"
                    f"**{report.gen_tps_target:,.0f}** tok/s\n\n"
                    f"*{report.gen_time_hours:.2f}h*"
                )
                kpi_train = (
                    f"### Train TPS\n\n"
                    f"**{report.train_tps_target:,.0f}** tok/s\n\n"
                    f"*{report.train_time_hours:.2f}h*"
                )
                kpi_bn = (
                    f"### Bottleneck\n\n"
                    f"**{report.bottleneck.title()}**\n\n"
                    f"*slack: {report.bottleneck_slack:.1%}*"
                )

                is_colocated = colocated == "Colocated"
                timeline_fig = build_timeline_figure(report, colocated=is_colocated)
                memory_fig = build_memory_figure(report)

                return (
                    gr.update(visible=True),  # results_container
                    kpi_epoch,
                    kpi_gen,
                    kpi_train,
                    kpi_bn,
                    timeline_fig,
                    memory_fig,
                    "",  # prediction_status
                )
            except Exception as e:
                return (
                    gr.update(visible=True),
                    "### Epoch Time\n\n--",
                    "### Gen TPS\n\n--",
                    "### Train TPS\n\n--",
                    "### Bottleneck\n\n--",
                    gr.update(),
                    gr.update(),
                    f"**Error:** {e}",
                )

        run_btn.click(
            fn=_run_prediction,
            inputs=_all_prediction_inputs,
            outputs=[
                res["results_container"],
                res["kpi_epoch"],
                res["kpi_gen_tps"],
                res["kpi_train_tps"],
                res["kpi_bottleneck"],
                res["timeline_plot"],
                res["memory_plot"],
                prediction_status,
            ],
        )

        # ======================================================
        # Wire: Search
        # ======================================================

        _search_inputs = (
            _model_inputs
            + _hw_inputs
            + _rl_inputs
            + [
                sc["mode"],
                sc["device_counts"],
                sc["optimization_target"],
                sc["sweep_param"],
                sc["sweep_values"],
            ]
        )

        def _run_search(
            # Model fields (22)
            m_name,
            m_hidden,
            m_vocab,
            m_layers,
            m_dtype,
            m_attn,
            m_ffn,
            m_residual,
            m_nheads,
            m_nkv,
            m_hdim,
            m_inter,
            m_kv_comp,
            m_q_comp,
            m_rope,
            m_window,
            m_nexp,
            m_topk,
            m_nshared,
            m_exp_inter,
            m_shared_inter,
            m_mhc,
            # Hardware fields (15)
            hw_name,
            total_dev,
            tp,
            pp,
            dp,
            ep,
            cp,
            cp_type,
            sp,
            zero_stage,
            pp_sched,
            recomp_attn,
            full_recomp,
            opt_offload,
            act_offload,
            # RL fields (14)
            total_prompts,
            group_size,
            avg_plen,
            avg_rlen,
            max_rlen,
            std_rlen,
            train_mbs,
            grad_acc,
            gen_bs,
            colocated,
            ref_model,
            ref_offload,
            spec_dec,
            mtp_len,
            # Search fields (5)
            search_mode,
            device_counts_str,
            opt_target,
            sweep_param,
            sweep_values_str,
        ):
            import plotly.graph_objects as _go

            try:
                model_cfg = _build_model_config(
                    m_name,
                    m_hidden,
                    m_vocab,
                    m_layers,
                    m_dtype,
                    m_attn,
                    m_ffn,
                    m_residual,
                    m_nheads,
                    m_nkv,
                    m_hdim,
                    m_inter,
                    m_kv_comp,
                    m_q_comp,
                    m_rope,
                    m_window,
                    m_nexp,
                    m_topk,
                    m_nshared,
                    m_exp_inter,
                    m_shared_inter,
                    m_mhc,
                )
                hw_cfg = _build_hw_config(hw_name)
                rl_cfg = _build_rl_config(
                    total_prompts,
                    group_size,
                    avg_plen,
                    avg_rlen,
                    max_rlen,
                    std_rlen,
                    train_mbs,
                    grad_acc,
                    gen_bs,
                    colocated,
                    ref_model,
                    ref_offload,
                    spec_dec,
                    mtp_len,
                )
                perf_model = RLPerformanceModel(model_cfg, hw_cfg)

                if search_mode == "Pareto Search":
                    counts = [
                        int(x.strip())
                        for x in device_counts_str.split(",")
                        if x.strip()
                    ]
                    search_results = pareto_search(perf_model, hw_cfg, rl_cfg, counts)

                    pareto_fig = build_pareto_figure(search_results)

                    # Build comparison dataframe
                    rows = []
                    for r in search_results:
                        tp_cfg = r.train_parallel
                        rows.append(
                            [
                                r.devices,
                                tp_cfg.tp,
                                tp_cfg.pp,
                                tp_cfg.dp,
                                tp_cfg.ep,
                                round(r.report.epoch_time_hours, 2),
                                round(r.report.gen_tps_target, 0),
                                round(r.report.train_tps_target, 0),
                                "Yes" if r.is_feasible else "No",
                                "Yes" if r.is_pareto else "",
                            ]
                        )

                    empty_sens = _go.Figure()
                    empty_sens.update_layout(
                        template="plotly_white", title="Sensitivity"
                    )

                    status = f"**Pareto search complete.** {len(search_results)} configs evaluated."
                    return pareto_fig, empty_sens, rows, status

                else:  # Sensitivity Analysis
                    values = [
                        int(x.strip()) for x in sweep_values_str.split(",") if x.strip()
                    ]
                    tp_v, pp_v, dp_v, ep_v, cp_v = (
                        int(tp),
                        int(pp),
                        int(dp),
                        int(ep),
                        int(cp),
                    )
                    total_dev_v = int(total_dev)

                    train_par = ParallelismConfig(
                        tp=tp_v,
                        pp=pp_v,
                        dp=dp_v,
                        ep=ep_v,
                        cp=cp_v,
                    )
                    gen_dp = total_dev_v // tp_v if tp_v > 0 else 1
                    gen_par = ParallelismConfig(tp=tp_v, pp=1, dp=gen_dp)

                    sweep_results = sensitivity_sweep(
                        perf_model,
                        hw_cfg,
                        rl_cfg,
                        param_name=sweep_param,
                        values=values,
                        total_devices=total_dev_v,
                        gen_parallel=gen_par,
                        train_parallel=train_par,
                    )

                    reports = [sr.report for sr in sweep_results]
                    sens_fig = build_sensitivity_figure(sweep_param, values, reports)

                    rows = []
                    for val, sr in zip(values, sweep_results):
                        rows.append(
                            [
                                sr.devices,
                                train_par.tp,
                                train_par.pp,
                                train_par.dp,
                                train_par.ep,
                                round(sr.report.epoch_time_hours, 2),
                                round(sr.report.gen_tps_target, 0),
                                round(sr.report.train_tps_target, 0),
                                "Yes" if sr.is_feasible else "No",
                                f"{sweep_param}={val}",
                            ]
                        )

                    empty_pareto = _go.Figure()
                    empty_pareto.update_layout(
                        template="plotly_white", title="Pareto Frontier"
                    )

                    status = f"**Sensitivity sweep complete.** {len(values)} values evaluated."
                    return empty_pareto, sens_fig, rows, status

            except Exception as e:
                empty1 = _go.Figure()
                empty1.update_layout(template="plotly_white")
                empty2 = _go.Figure()
                empty2.update_layout(template="plotly_white")
                return empty1, empty2, [], f"**Error:** {e}"

        sc["search_btn"].click(
            fn=_run_search,
            inputs=_search_inputs,
            outputs=[
                sc["pareto_plot"],
                sc["sens_plot"],
                sc["comparison_df"],
                sc["search_status"],
            ],
        )

    return app


def launch(host: str = "127.0.0.1", port: int = 7860, share: bool = False):
    """Create and launch the Gradio app."""
    app = create_app()
    app.launch(
        server_name=host,
        server_port=port,
        share=share,
        theme=gr.themes.Soft(),
    )

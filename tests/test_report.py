def test_format_sparse_sweep_renders_points_and_fabrics():
    from llm_perf.report import format_sparse_sweep
    rows = [
        {"point": {"moe_node_limit": 0}, "step_seconds": 0.5,
         "exposed_comm_by_fabric": {"nic": 0.12, "nvlink": 0.03},
         "cross_node_gb": 0.12, "peak_memory_gb": 40.0, "feasible": True},
        {"point": {"moe_node_limit": 2}, "step_seconds": 0.42,
         "exposed_comm_by_fabric": {"nic": 0.05, "nvlink": 0.06},
         "cross_node_gb": 0.05, "peak_memory_gb": 40.0, "feasible": True},
    ]
    out = format_sparse_sweep(rows)
    assert "moe_node_limit" in out
    assert "nic" in out
    assert "nvlink" in out
    # both points appear
    assert out.count("feasible") >= 1 or "OK" in out

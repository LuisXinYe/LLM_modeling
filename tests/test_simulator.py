import pytest
from llm_perf.simulator import simulate, SimResult
from llm_perf.builder import SimOp


def test_empty_ops():
    r = simulate([])
    assert r.wall_clock_time == 0


def test_single_op():
    ops = [SimOp("op0", "compute", duration=1.0, depends_on=[], weight_bytes=100, output_bytes=50)]
    r = simulate(ops)
    assert r.wall_clock_time == pytest.approx(1.0)
    assert r.weight_bytes == 100
    assert r.peak_activation_bytes >= 50


def test_sequential_same_stream():
    ops = [
        SimOp("op0", "compute", 1.0, []),
        SimOp("op1", "compute", 2.0, [0]),
    ]
    r = simulate(ops)
    assert r.wall_clock_time == pytest.approx(3.0)


def test_parallel_different_streams():
    ops = [
        SimOp("op0", "compute", 2.0, []),
        SimOp("op1", "tp_comm", 1.0, []),  # no dependency, parallel
    ]
    r = simulate(ops)
    assert r.wall_clock_time == pytest.approx(2.0)  # max of parallel


def test_dependency_across_streams():
    ops = [
        SimOp("compute0", "compute", 2.0, []),
        SimOp("comm0", "tp_comm", 1.0, [0]),    # depends on compute0
        SimOp("compute1", "compute", 1.0, [1]),  # depends on comm0
    ]
    r = simulate(ops)
    # compute0: 0→2, comm0: 2→3, compute1: 3→4
    assert r.wall_clock_time == pytest.approx(4.0)


def test_overlap_compute_and_comm():
    ops = [
        SimOp("compute0", "compute", 3.0, []),
        SimOp("comm0", "tp_comm", 2.0, [0]),     # after compute0 on comm stream
        SimOp("compute1", "compute", 3.0, [0]),   # after compute0 on compute stream
    ]
    r = simulate(ops)
    # compute0: 0→3, comm0: 3→5 (tp_comm), compute1: 3→6 (compute)
    assert r.wall_clock_time == pytest.approx(6.0)


def test_comm_hidden_by_longer_compute():
    ops = [
        SimOp("compute0", "compute", 1.0, []),
        SimOp("comm0", "tp_comm", 1.0, [0]),      # 1→2 on comm
        SimOp("compute1", "compute", 3.0, [0]),    # 1→4 on compute
        SimOp("compute2", "compute", 1.0, [1, 2]), # needs both: max(2,4)=4, so 4→5
    ]
    r = simulate(ops)
    assert r.wall_clock_time == pytest.approx(5.0)


def test_memory_peak_tracking():
    ops = [
        SimOp("op0", "compute", 1.0, [], weight_bytes=0, output_bytes=100),
        SimOp("op1", "compute", 1.0, [0], weight_bytes=0, output_bytes=200),
        SimOp("op2", "compute", 1.0, [1], weight_bytes=0, output_bytes=50),
    ]
    r = simulate(ops)
    # At op1: op0 output(100) + op1 output(200) = 300 (before op0 freed)
    assert r.peak_activation_bytes >= 200  # At least op1's output


def test_memory_freed_after_consumed():
    ops = [
        SimOp("op0", "compute", 1.0, [], output_bytes=1000),
        SimOp("op1", "compute", 1.0, [0], output_bytes=100),
        SimOp("op2", "compute", 1.0, [1], output_bytes=100),
    ]
    r = simulate(ops)
    # op0's 1000 bytes should be freed after op1 consumes it
    # Peak should NOT be 1000 + 100 + 100 = 1200
    # Peak is 1000 + 100 = 1100 (at op1 finish, before op0 freed)
    assert r.peak_activation_bytes <= 1200


def test_weight_bytes_accumulated():
    ops = [
        SimOp("fwd", "compute", 1.0, [], weight_bytes=500),
        SimOp("bwd", "compute", 1.0, [0], weight_bytes=500),
    ]
    r = simulate(ops)
    assert r.weight_bytes >= 500


def test_result_is_simresult():
    r = simulate([])
    assert isinstance(r, SimResult)


def test_stream_clock_independence():
    """Two streams run independently; their ops do not block each other."""
    ops = [
        SimOp("a0", "stream_a", 5.0, []),
        SimOp("a1", "stream_a", 5.0, [0]),  # stream_a: 0→5→10
        SimOp("b0", "stream_b", 1.0, []),   # stream_b: 0→1  (no deps on stream_a)
    ]
    r = simulate(ops)
    assert r.wall_clock_time == pytest.approx(10.0)


def test_diamond_dependency():
    """Diamond DAG: two parallel ops both depend on one source and converge."""
    ops = [
        SimOp("src", "compute", 1.0, []),           # 0→1
        SimOp("left", "compute", 2.0, [0]),          # 1→3
        SimOp("right", "tp_comm", 1.0, [0]),         # 1→2 (tp_comm stream)
        SimOp("sink", "compute", 1.0, [1, 2]),       # max(3,2)=3 → 4
    ]
    r = simulate(ops)
    assert r.wall_clock_time == pytest.approx(4.0)


def test_same_fabric_collectives_serialize():
    # Two independent NIC collectives (no dep) must time-share the NIC:
    # wall on nic = sum of durations, not max.
    ops = [
        SimOp(name="dp", stream="dp_comm", duration=1.0, depends_on=[], fabric="nic"),
        SimOp(name="ep", stream="ep_comm", duration=1.0, depends_on=[], fabric="nic"),
    ]
    res = simulate(ops)
    assert res.wall_clock_time == pytest.approx(2.0)


def test_different_fabric_collectives_overlap():
    # NVLink (intra) + NIC (inter), independent → run in parallel.
    ops = [
        SimOp(name="tp", stream="tp_comm", duration=1.0, depends_on=[], fabric="nvlink"),
        SimOp(name="dp", stream="dp_comm", duration=1.0, depends_on=[], fabric="nic"),
    ]
    res = simulate(ops)
    assert res.wall_clock_time == pytest.approx(1.0)


def test_comm_still_overlaps_compute():
    # A NIC collective independent of a compute op overlaps it (fabric != compute).
    ops = [
        SimOp(name="bwd", stream="compute", duration=2.0, depends_on=[], fabric=None),
        SimOp(name="dp", stream="dp_comm", duration=1.0, depends_on=[], fabric="nic"),
    ]
    res = simulate(ops)
    assert res.wall_clock_time == pytest.approx(2.0)  # comm hidden under compute


def test_exposed_comm_by_fabric_reported():
    ops = [
        SimOp(name="bwd", stream="compute", duration=0.5, depends_on=[], fabric=None),
        SimOp(name="dp", stream="dp_comm", duration=2.0, depends_on=[], fabric="nic"),
    ]
    res = simulate(ops)
    # 2.0 nic comm, only 0.5 hidden → 1.5 exposed on nic
    assert res.exposed_comm_by_fabric["nic"] == pytest.approx(1.5)

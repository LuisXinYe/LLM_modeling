from pathlib import Path
import pytest
from llm_perf.config import (
    ParallelismConfig, WorkloadConfig, load_hardware_config, load_model_config,
)
from llm_perf.builder import _split_stages
from llm_perf.pp_pipeline import (
    stage_unit_time,
    pipeline_schedule,
    simulate_pipeline,
    PipelineResult,
)

CONFIGS = Path(__file__).parent.parent / "configs"

@pytest.fixture
def mc():
    return load_model_config(str(CONFIGS / "models" / "llama3_1_8b.yaml"))

@pytest.fixture
def hw():
    return load_hardware_config(str(CONFIGS / "hardware" / "ascend_910c.yaml"))

def test_stage_unit_time_overlap_and_bwd(mc, hw):
    par = ParallelismConfig(tp=2, cp=8, dp=1, pp=1)
    wl = WorkloadConfig(group_size=1)
    chunk = _split_stages(mc.get_layers(), 8)[0]  # one PP stage worth of layers
    fwd_t, bwd_t = stage_unit_time(mc, hw, par, wl, chunk, chunk_id=0, cp=8, seq_len=32768)
    assert fwd_t > 0
    assert bwd_t == pytest.approx(2.0 * fwd_t)  # default bwd_factor

def test_stage_unit_time_cp1_no_ring_overlap_noop(mc, hw):
    # cp=1 has no CP ring comm, so overlap max(compute, cp_comm) == compute
    par = ParallelismConfig(tp=2, cp=1, dp=1, pp=1)
    wl = WorkloadConfig(group_size=1)
    chunk = _split_stages(mc.get_layers(), 8)[0]
    fwd_t, _ = stage_unit_time(mc, hw, par, wl, chunk, chunk_id=0, cp=1, seq_len=4096)
    # fwd_t equals compute + tp_comm (cp_comm == 0): recompute the raw sim to compare
    from llm_perf.builder import build_forward_pass
    from llm_perf.simulator import simulate
    p1 = par.model_copy(update={"cp": 1, "pp": 1})
    w1 = wl.model_copy(update={"avg_prompt_len": 4096, "avg_response_len": 0,
                               "train_micro_batch_size": 1})
    sim = simulate(build_forward_pass(mc, hw, p1, w1, stage_layers=chunk))
    assert fwd_t == pytest.approx(sim.compute_time + sim.tp_comm_time)


def test_schedule_counts_v1():
    m, p = 6, 4
    sched = pipeline_schedule(m, p, v=1)
    assert len(sched) == p
    for d in range(p):
        evs = sched[d]
        # each device runs m forwards and m backwards
        assert sum(1 for _, _, ph in evs if ph == "F") == m
        assert sum(1 for _, _, ph in evs if ph == "B") == m
        # all events on device d belong to vstage d (V=1)
        assert all(vs == d for _, vs, _ in evs)
        # 1F1B warmup: device d issues (p-1-d) forwards, then one more F before
        # its first backward (F always precedes its matching B in steady state)
        first_b = next(i for i, (_, _, ph) in enumerate(evs) if ph == "B")
        assert first_b == (p - d)


def test_schedule_counts_v2():
    m, p, v = 4, 4, 2
    sched = pipeline_schedule(m, p, v)
    assert len(sched) == p
    for d in range(p):
        evs = sched[d]
        # v virtual stages per device, each with m F and m B
        assert sum(1 for _, _, ph in evs if ph == "F") == m * v
        assert sum(1 for _, _, ph in evs if ph == "B") == m * v
        assert all(vs % p == d for _, vs, _ in evs)


def test_schedule_forward_precedes_backward_per_unit():
    # Causality invariant: for every (unit_idx, vstage) pair on a device,
    # the F event must appear before the matching B event in that device's
    # ordered event list.
    for m, p, v in [(2, 4, 1), (4, 4, 2)]:
        sched = pipeline_schedule(m, p, v)
        for d, evs in enumerate(sched):
            f_pos = {}
            for pos, (unit_idx, vstage, phase) in enumerate(evs):
                key = (unit_idx, vstage)
                if phase == "F":
                    f_pos[key] = pos
                else:  # phase == "B"
                    assert key in f_pos, (
                        f"device {d}: B for {key} occurs with no prior F "
                        f"(m={m}, p={p}, v={v})"
                    )
                    assert f_pos[key] < pos, (
                        f"device {d}: F for {key} does not precede its B "
                        f"(m={m}, p={p}, v={v})"
                    )


def _equal_times(m, p, v, fwd=1.0, bwd=2.0):
    return [[(fwd, bwd)] * (p * v) for _ in range(m)]


def test_anchor_bubble_v1():
    m, p = 8, 4
    res = simulate_pipeline(_equal_times(m, p, 1), [1.0] * m, p, v=1)
    assert isinstance(res, PipelineResult)
    expected = (p - 1) / (m + p - 1)         # standard 1F1B bubble
    assert res.bubble_ratio == pytest.approx(expected, abs=0.02)


def test_simulate_pipeline_v2_not_implemented():
    m, p, v = 8, 4, 2
    with pytest.raises(NotImplementedError):
        simulate_pipeline(_equal_times(m, p, v), [1.0] * m, p, v=v)


def test_pp1_no_bubble():
    m, p = 5, 1
    res = simulate_pipeline(_equal_times(m, p, 1, fwd=1.0, bwd=2.0), [1.0] * m, p, v=1)
    assert res.bubble_ratio == pytest.approx(0.0, abs=1e-9)
    assert res.step_time == pytest.approx(m * (1.0 + 2.0))


def test_more_microbatches_smaller_bubble():
    p = 4
    b_few = simulate_pipeline(_equal_times(4, p, 1), [1.0] * 4, p, v=1).bubble_ratio
    b_many = simulate_pipeline(_equal_times(16, p, 1), [1.0] * 16, p, v=1).bubble_ratio
    assert b_many < b_few


def _unequal_times(m, p):
    # Unequal per-unit fwd/bwd durations, replicated across all p devices
    # for each unit (v=1 -> S=p entries per unit).
    fwds = [1.0 + 0.3 * ((j * 7) % 5) for j in range(m)]
    bwds = [2.0 + 0.4 * ((j * 5) % 4) for j in range(m)]
    return [[(fwds[j], bwds[j])] * p for j in range(m)]


def test_1f1b_memory_bounded_variable_durations():
    # 1F1B's defining memory property: in-flight activations are capped by
    # pipeline depth p, independent of the number of microbatches m. With
    # unequal per-unit durations, the old greedy/FIFO-tie-break schedule
    # degraded to an eager order whose peak activation grows with m; hard
    # schedule-order edges restore the bound.
    p = 4
    m_small, m_large = 8, 16
    res_small = simulate_pipeline(
        _unequal_times(m_small, p), [1.0] * m_small, p, v=1
    )
    res_large = simulate_pipeline(
        _unequal_times(m_large, p), [1.0] * m_large, p, v=1
    )
    assert res_large.peak_activation_bytes <= 1.5 * res_small.peak_activation_bytes


def test_variable_length_bubble_differs_from_closed_form():
    # mix of fast (short) and slow (long) units → bubble != equal-length formula
    p, v = 4, 1
    times = []
    for j in range(8):
        t = 1.0 if j % 2 == 0 else 4.0   # alternating fast/slow units
        times.append([(t, 2.0 * t)] * (p * v))
    res = simulate_pipeline(times, [1.0] * 8, p, v=v)
    closed = (p - 1) / (8 + p - 1)
    assert res.bubble_ratio != pytest.approx(closed, abs=1e-6)
    assert 0.0 <= res.bubble_ratio < 1.0


def test_ordering_affects_bubble():
    # clustering all slow units last vs interleaving changes the bubble
    p, v = 4, 1
    def mk(order):
        return [[(t, 2.0 * t)] * (p * v) for t in order]
    clustered = mk([1, 1, 1, 1, 4, 4, 4, 4])
    interleaved = mk([1, 4, 1, 4, 1, 4, 1, 4])
    b_cluster = simulate_pipeline(clustered, [1.0] * 8, p, v=v).bubble_ratio
    b_inter = simulate_pipeline(interleaved, [1.0] * 8, p, v=v).bubble_ratio
    assert b_cluster != pytest.approx(b_inter, abs=1e-6)

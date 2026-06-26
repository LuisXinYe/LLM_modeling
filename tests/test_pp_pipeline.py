from pathlib import Path
import pytest
from llm_perf.config import (
    ParallelismConfig, WorkloadConfig, load_hardware_config, load_model_config,
)
from llm_perf.builder import _split_stages
from llm_perf.pp_pipeline import stage_unit_time

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

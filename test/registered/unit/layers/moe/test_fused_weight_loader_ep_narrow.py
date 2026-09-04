import torch

from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _bare_layer(num_global: int, num_local: int, storage_rank: int) -> FusedMoE:
    layer = FusedMoE.__new__(FusedMoE)
    torch.nn.Module.__init__(layer)
    layer._num_global_routed = num_global
    layer._num_local_routed = num_local
    layer._expert_storage_rank = storage_rank
    return layer


def test_narrows_global_tensor_to_this_ranks_experts():
    layer = _bare_layer(num_global=8, num_local=2, storage_rank=3)
    weight = torch.arange(8 * 3 * 4).reshape(8, 3, 4)

    out = layer._narrow_fused_weight_to_local_experts(weight)

    assert torch.equal(out, weight[6:8])


def test_narrows_bias_tensor_on_dim_zero():
    layer = _bare_layer(num_global=8, num_local=4, storage_rank=1)
    bias = torch.arange(8 * 5).reshape(8, 5)

    out = layer._narrow_fused_weight_to_local_experts(bias)

    assert torch.equal(out, bias[4:8])


def test_passes_through_pre_sliced_tensor():
    layer = _bare_layer(num_global=8, num_local=2, storage_rank=3)
    weight = torch.zeros(2, 3, 4)

    assert layer._narrow_fused_weight_to_local_experts(weight) is weight


def test_passes_through_without_expert_parallelism():
    layer = _bare_layer(num_global=8, num_local=8, storage_rank=0)
    weight = torch.zeros(8, 3, 4)

    assert layer._narrow_fused_weight_to_local_experts(weight) is weight


def test_passes_through_helper_without_expert_bookkeeping():
    layer = FusedMoE.__new__(FusedMoE)
    torch.nn.Module.__init__(layer)
    weight = torch.zeros(8, 3, 4)

    assert layer._narrow_fused_weight_to_local_experts(weight) is weight

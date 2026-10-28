import pytest

from kitsune.cluster import Partition
from kitsune.config import MIN_CONTEXT
from kitsune.estimate import (
    ModelInfo,
    dtype_for,
    parse_model_info,
    required_gib,
    solve,
)

QWEN_05B_API = {"safetensors": {"parameters": {"BF16": 494032768}, "total": 494032768}}
QWEN_05B_CONFIG = {
    "num_hidden_layers": 24,
    "num_attention_heads": 14,
    "num_key_value_heads": 2,
    "hidden_size": 896,
    "max_position_embeddings": 32768,
}
TINY = parse_model_info("Qwen/Qwen2.5-0.5B-Instruct", QWEN_05B_API, QWEN_05B_CONFIG)


def part(name, gpu_type, *, max_node_gpus=1, max_node_free=1, free=1, total=1, pending=0, down=0):
    return Partition(
        name=name,
        gpu_type=gpu_type,
        is_default=False,
        nodes=1 + down,
        nodes_down=down,
        gpus_total=total,
        gpus_free=free,
        max_node_gpus=max_node_gpus,
        max_node_free=max_node_free,
        pending=pending,
    )


def big(weights_gib, *, layers, heads, kv_heads, head_dim, native=32768):
    return ModelInfo(
        repo="x/big",
        weight_bytes=int(weights_gib * 1024**3),
        layers=layers,
        heads=heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        native_max_len=native,
    )


def test_parse_model_info_reads_weights_and_kv_shape():
    assert TINY.weight_bytes == 494032768 * 2
    assert TINY.layers == 24
    assert TINY.heads == 14
    assert TINY.kv_heads == 2
    assert TINY.head_dim == 64  # 896 / 14
    assert TINY.kv_bytes_per_token == 2 * 24 * 2 * 64 * 2


def test_parse_model_info_reads_native_context():
    assert TINY.native_max_len == 32768


def test_parse_model_info_without_native_context_falls_back_to_none():
    config = {key: value for key, value in QWEN_05B_CONFIG.items() if key != "max_position_embeddings"}
    info = parse_model_info("a/b", QWEN_05B_API, config)
    assert info.native_max_len is None


def test_parse_model_info_without_safetensors_is_an_error():
    with pytest.raises(ValueError, match="safetensors"):
        parse_model_info("a/b", {}, QWEN_05B_CONFIG)


def test_parse_model_info_missing_config_field_is_an_error():
    with pytest.raises(ValueError, match="num_hidden_layers"):
        parse_model_info("a/b", QWEN_05B_API, {"num_attention_heads": 14, "hidden_size": 896})


def test_required_memory_drops_with_tensor_parallel():
    one = required_gib(TINY, 1, 4096)
    two = required_gib(TINY, 2, 4096)
    assert two < one
    assert one > 2.0  # pelo menos o overhead


def test_auto_plan_serves_native_context_with_derived_util():
    plans, rejected = solve(TINY, [part("l40s", "L40S", max_node_gpus=4, max_node_free=4)])
    plan = plans[0]
    assert plan.gpus == 1
    assert plan.max_len == 32768
    assert plan.full_context
    # pesos ~0.92 + KV 0.375 + overhead 2 = ~3.3 GiB; 3.3/45 = 0.0733 -> 0.08
    assert plan.gpu_util == 0.08


def test_auto_plan_falls_back_to_min_context_without_native():
    info = parse_model_info(
        "a/b",
        QWEN_05B_API,
        {key: value for key, value in QWEN_05B_CONFIG.items() if key != "max_position_embeddings"},
    )
    plans, rejected = solve(info, [part("l40s", "L40S")])
    assert plans[0].max_len == MIN_CONTEXT
    assert plans[0].full_context


def test_explicit_max_len_becomes_the_target():
    plans, rejected = solve(TINY, [part("l40s", "L40S")], max_len=8192)
    assert plans[0].max_len == 8192
    assert plans[0].full_context


def test_explicit_gpu_util_caps_what_fits():
    model = big(30, layers=40, heads=32, kv_heads=8, head_dim=128)
    # precisa (30+10)/tp + 2 por GPU: com teto 0.3 só cabe com 4 GPUs (0.27)
    plans, rejected = solve(model, [part("l40s", "L40S", max_node_gpus=4, max_node_free=4)], gpu_util=0.3)
    assert plans[0].gpus == 4
    assert plans[0].gpu_util == 0.3  # override é usado na linha de comando, não o derivado


def test_auto_ceiling_fits_model_on_one_gpu_with_high_util():
    model = big(30, layers=40, heads=32, kv_heads=8, head_dim=128)
    # (30+5)/1 + 2 = 37 GiB numa L40S: cabe com 1 GPU (util derivado 0.83);
    # o teto antigo de 0.3 pediria 4 GPUs para o mesmo modelo
    plans, rejected = solve(model, [part("l40s", "L40S", max_node_gpus=4, max_node_free=4)])
    assert plans[0].gpus == 1
    assert plans[0].gpu_util == 0.83


def test_context_is_clamped_with_a_clear_flag_when_native_does_not_fit():
    chatty = big(0.92, layers=24, heads=14, kv_heads=2, head_dim=64, native=2_000_000)
    plans, rejected = solve(chatty, [part("rtx5000", "RTX5000")])
    plan = plans[0]
    assert not plan.full_context
    assert MIN_CONTEXT < plan.max_len < 2_000_000


def test_tensor_parallel_must_divide_attention_heads():
    # 14 cabeças: TP 4 não é válido; com 8 GPUs no nó os candidatos são 1 e 2
    plans, rejected = solve(TINY, [part("big", "L40S", max_node_gpus=8, max_node_free=8)])
    assert plans[0].gpus == 1  # cabe com 1, então nem chega a pedir mais


def test_explicit_gpus_that_do_not_divide_heads_are_rejected():
    plans, rejected = solve(TINY, [part("l40s", "L40S", max_node_gpus=8)], gpus=4)
    assert plans == []
    assert "não divide" in rejected[0].reason


def test_explicit_gpus_above_the_node_are_rejected():
    plans, rejected = solve(TINY, [part("l40s", "L40S", max_node_gpus=4)], gpus=8)
    assert plans == []
    assert "no máximo 4" in rejected[0].reason


def test_explicit_max_len_that_does_not_fit_reports_how_much_would():
    plans, rejected = solve(TINY, [part("rtx5000", "RTX5000")], max_len=2_000_000)
    assert plans == []
    assert "cabem" in rejected[0].reason
    assert "2000000" in rejected[0].reason


def test_small_model_prefers_partitions_with_free_gpus():
    partitions = [
        part("l40s", "L40S", max_node_gpus=4, max_node_free=0, free=0, total=8, pending=6),
        part("rtx8000", "RTX8000", max_node_gpus=8, max_node_free=2, free=2, total=16),
    ]
    plans, rejected = solve(TINY, partitions)
    assert [plan.partition for plan in plans] == ["rtx8000", "l40s"]
    assert plans[0].gpus == 1
    assert plans[0].dtype == "half"  # Turing (7.5) não tem bf16
    assert plans[1].dtype == "auto"  # Ada (8.9) tem


def test_pascal_is_rejected_because_vllm_needs_7_or_newer():
    plans, rejected = solve(TINY, [part("p5000", "P5000", max_node_free=5)])
    assert plans == []
    assert rejected[0].partition == "p5000"
    assert "6.1" in rejected[0].reason


def test_partition_with_all_nodes_down_is_rejected():
    down = part("h100", "H100", max_node_gpus=0, max_node_free=0, free=0, total=0, down=1)
    plans, rejected = solve(TINY, [down])
    assert plans == []
    assert "fora do ar" in rejected[0].reason


def test_too_big_reports_why_it_does_not_fit():
    model = big(400, layers=80, heads=64, kv_heads=8, head_dim=128)
    plans, rejected = solve(model, [part("rtx8000", "RTX8000", max_node_gpus=8, max_node_free=8, free=8, total=8)])
    assert plans == []
    assert "nem o mínimo" in rejected[0].reason


def test_dtype_for_compute_capability():
    assert dtype_for(7.5) == "half"
    assert dtype_for(8.0) == "auto"
    assert dtype_for(8.9) == "auto"

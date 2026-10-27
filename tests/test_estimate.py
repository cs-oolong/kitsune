import pytest

from kitsune.cluster import Partition
from kitsune.estimate import (
    ModelInfo,
    dtype_for,
    parse_model_info,
    required_gib,
    suggest,
)

QWEN_05B_API = {"safetensors": {"parameters": {"BF16": 494032768}, "total": 494032768}}
QWEN_05B_CONFIG = {
    "num_hidden_layers": 24,
    "num_attention_heads": 14,
    "num_key_value_heads": 2,
    "hidden_size": 896,
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


def test_parse_model_info_reads_weights_and_kv_shape():
    assert TINY.weight_bytes == 494032768 * 2
    assert TINY.layers == 24
    assert TINY.heads == 14
    assert TINY.kv_heads == 2
    assert TINY.head_dim == 64  # 896 / 14
    assert TINY.kv_bytes_per_token == 2 * 24 * 2 * 64 * 2


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


def test_tensor_parallel_must_divide_attention_heads():
    # 14 cabeças: TP 4 não é válido, então o maior TP possível com 8 GPUs é 2
    suggestions, rejected = suggest(
        TINY,
        [part("big", "L40S", max_node_gpus=8, max_node_free=8)],
        gpu_util=0.3,
        max_len=4096,
    )
    assert suggestions[0].gpus == 1  # cabe com 1, então nem chega a TP 4


def test_small_model_fits_on_one_gpu_and_prefers_free_partitions():
    partitions = [
        part("l40s", "L40S", max_node_gpus=4, max_node_free=0, free=0, total=8, pending=6),
        part("rtx8000", "RTX8000", max_node_gpus=8, max_node_free=2, free=2, total=16),
    ]
    suggestions, rejected = suggest(TINY, partitions, gpu_util=0.3, max_len=4096)
    assert [s.partition for s in suggestions] == ["rtx8000", "l40s"]
    assert suggestions[0].gpus == 1
    assert suggestions[0].dtype == "half"  # Turing (7.5) não tem bf16
    assert suggestions[1].dtype == "auto"  # Ada (8.9) tem


def test_pascal_is_rejected_because_vllm_needs_7_or_newer():
    suggestions, rejected = suggest(TINY, [part("p5000", "P5000", max_node_free=5)], gpu_util=0.3, max_len=4096)
    assert suggestions == []
    assert rejected[0].partition == "p5000"
    assert "6.1" in rejected[0].reason


def test_partition_with_all_nodes_down_is_rejected():
    down = part("h100", "H100", max_node_gpus=0, max_node_free=0, free=0, total=0, down=1)
    suggestions, rejected = suggest(TINY, [down], gpu_util=0.3, max_len=4096)
    assert suggestions == []
    assert "fora do ar" in rejected[0].reason


def test_model_too_big_for_one_gpu_asks_for_more_gpus():
    big = ModelInfo(
        repo="x/big",
        weight_bytes=int(30 * 1024**3),  # ~30 GiB de pesos
        layers=40,
        heads=32,
        kv_heads=8,
        head_dim=128,
    )
    suggestions, rejected = suggest(
        big,
        [part("l40s", "L40S", max_node_gpus=4, max_node_free=4, free=4, total=4)],
        gpu_util=0.3,
        max_len=4096,
    )
    # 45 GiB * 0.3 = 13.5 GiB utilizáveis: precisa de pelo menos 4 GPUs
    assert suggestions[0].gpus == 4


def test_too_big_reports_why_it_does_not_fit():
    big = ModelInfo(
        repo="x/big",
        weight_bytes=int(200 * 1024**3),
        layers=80,
        heads=64,
        kv_heads=8,
        head_dim=128,
    )
    suggestions, rejected = suggest(
        big,
        [part("rtx8000", "RTX8000", max_node_gpus=8, max_node_free=8, free=8, total=8)],
        gpu_util=0.3,
        max_len=4096,
    )
    assert suggestions == []
    assert "não cabe" in rejected[0].reason


def test_dtype_for_compute_capability():
    assert dtype_for(7.5) == "half"
    assert dtype_for(8.0) == "auto"
    assert dtype_for(8.9) == "auto"

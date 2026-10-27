"""Quanto de GPU um modelo precisa no vLLM, e em qual partição do cluster cabe."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

from .cluster import Partition
from .config import GPU_SPECS, MIN_COMPUTE_CAPABILITY

GIB = 1024**3
OVERHEAD_GIB = 2.0
TP_CHOICES = (1, 2, 4, 8)
BF16_MIN_COMPUTE_CAPABILITY = 8.0
BYTES_PER_PARAM = {
    "F64": 8,
    "I64": 8,
    "F32": 4,
    "I32": 4,
    "BF16": 2,
    "F16": 2,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I4": 0.5,
}


@dataclass(frozen=True)
class ModelInfo:
    repo: str
    weight_bytes: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int

    @property
    def kv_bytes_per_token(self) -> int:
        return 2 * self.layers * self.kv_heads * self.head_dim * 2


@dataclass(frozen=True)
class Suggestion:
    partition: str
    gpu_type: str
    gpus: int
    need_gib: float
    usable_gib: float
    dtype: str
    free_now: int
    pending: int


@dataclass(frozen=True)
class Rejection:
    partition: str
    reason: str


def parse_model_info(repo: str, api: dict, config: dict) -> ModelInfo:
    safetensors = api.get("safetensors") or {}
    parameters = safetensors.get("parameters") or {}
    if parameters:
        weight_bytes = sum(int(count) * BYTES_PER_PARAM.get(dtype, 2) for dtype, count in parameters.items())
    else:
        weight_bytes = int(safetensors.get("total", 0)) * 2
    if weight_bytes == 0:
        raise ValueError(f"{repo}: não achei o tamanho dos pesos (o repositório tem safetensors?)")
    try:
        heads = int(config["num_attention_heads"])
        return ModelInfo(
            repo=repo,
            weight_bytes=weight_bytes,
            layers=int(config["num_hidden_layers"]),
            heads=heads,
            kv_heads=int(config.get("num_key_value_heads", heads)),
            head_dim=int(config.get("head_dim") or int(config["hidden_size"]) // heads),
        )
    except KeyError as error:
        raise ValueError(f"{repo}: config.json sem o campo {error}") from error


def _get_json(url: str, token: str | None) -> dict:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise ValueError(f"Hugging Face respondeu {error.code} para {url}") from error
    except OSError as error:
        raise ValueError(f"não consegui consultar o Hugging Face: {error}") from error


def fetch_model_info(repo: str) -> ModelInfo:
    token = os.environ.get("HF_TOKEN")
    api = _get_json(f"https://huggingface.co/api/models/{repo}", token)
    config = _get_json(f"https://huggingface.co/{repo}/resolve/main/config.json", token)
    return parse_model_info(repo, api, config)


def weight_gib(info: ModelInfo) -> float:
    return info.weight_bytes / GIB


def kv_gib(info: ModelInfo, max_len: int) -> float:
    return info.kv_bytes_per_token * max_len / GIB


def required_gib(info: ModelInfo, tensor_parallel: int, max_len: int) -> float:
    per_gpu = (info.weight_bytes + info.kv_bytes_per_token * max_len) / tensor_parallel
    return per_gpu / GIB + OVERHEAD_GIB


def dtype_for(compute_capability: float) -> str:
    return "half" if compute_capability < BF16_MIN_COMPUTE_CAPABILITY else "auto"


def suggest(
    info: ModelInfo,
    partitions: list[Partition],
    *,
    gpu_util: float,
    max_len: int,
) -> tuple[list[Suggestion], list[Rejection]]:
    suggestions: list[Suggestion] = []
    rejections: list[Rejection] = []
    for part in partitions:
        if part.gpu_type is None:
            continue
        spec = GPU_SPECS.get(part.gpu_type)
        if spec is None:
            rejections.append(Rejection(part.name, f"tipo de GPU {part.gpu_type} sem especificação cadastrada"))
            continue
        mem_gib, capability = spec
        if part.max_node_gpus == 0:
            rejections.append(Rejection(part.name, "nenhum nó utilizável (fora do ar)"))
            continue
        if capability < MIN_COMPUTE_CAPABILITY:
            rejections.append(
                Rejection(part.name, f"compute capability {capability}; o vLLM precisa de {MIN_COMPUTE_CAPABILITY} ou mais")
            )
            continue

        usable = mem_gib * gpu_util
        valid = [tp for tp in TP_CHOICES if tp <= part.max_node_gpus and info.heads % tp == 0]
        if not valid:
            rejections.append(
                Rejection(
                    part.name,
                    f"nenhum paralelismo de tensor com até {part.max_node_gpus} GPU(s) divide as {info.heads} cabeças de atenção",
                )
            )
            continue
        fit = next((tp for tp in valid if required_gib(info, tp, max_len) <= usable), None)
        if fit is None:
            need = required_gib(info, valid[-1], max_len)
            rejections.append(
                Rejection(
                    part.name,
                    f"não cabe nem com {valid[-1]} GPU(s): precisa de {need:.1f} GiB por GPU, "
                    f"a GPU tem {usable:.1f} GiB utilizáveis com gpu-util {gpu_util}",
                )
            )
            continue
        suggestions.append(
            Suggestion(
                partition=part.name,
                gpu_type=part.gpu_type,
                gpus=fit,
                need_gib=required_gib(info, fit, max_len),
                usable_gib=usable,
                dtype=dtype_for(capability),
                free_now=part.max_node_free,
                pending=part.pending,
            )
        )

    suggestions.sort(key=lambda s: (s.free_now < s.gpus, s.gpus, s.pending, s.partition))
    return suggestions, rejections

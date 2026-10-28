"""Quanto de GPU um modelo precisa no vLLM, e qual configuração pedir ao Slurm.

O solucionador (`solve`) cruza o modelo (tamanho dos pesos, formato das camadas
e contexto nativo, lidos do Hugging Face) com o inventário do cluster e devolve
planos completos: partição, número de GPUs, gpu-util derivado e comprimento de
contexto. As flags de `start`/`suggest` são overrides sobre essa derivação.
"""

from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

from .cluster import Partition
from .config import GPU_SPECS, GPU_UTIL_CEILING, MIN_COMPUTE_CAPABILITY, MIN_CONTEXT

GIB = 1024**3
# Memória que o vLLM gasta além dos pesos (ativações do profiling, CUDA graphs, buffers).
# Medido no cluster com o vLLM 0.11: ~7.7 GiB para o gpt-oss-20b (2026-10-28); era 2 GiB
# na era do vLLM 0.8.5, que compilava bem menos.
OVERHEAD_GIB = 8.0
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
    # max_position_embeddings do config.json; None se o campo não existir.
    native_max_len: int | None

    @property
    def kv_bytes_per_token(self) -> int:
        return 2 * self.layers * self.kv_heads * self.head_dim * 2


@dataclass(frozen=True)
class Plan:
    partition: str
    gpu_type: str
    gpus: int
    gpu_util: float
    max_len: int
    dtype: str
    need_gib: float
    fits_now: bool
    free_now: int
    pending: int
    # True quando o plano serve o contexto alvo inteiro (o nativo, ou o pedido).
    full_context: bool


@dataclass(frozen=True)
class Rejection:
    partition: str
    reason: str


def parse_model_info(repo: str, api: dict, config: dict) -> ModelInfo:
    safetensors = api.get("safetensors") or {}
    parameters = safetensors.get("parameters") or {}
    bytes_per_param = BYTES_PER_PARAM
    # MXFP4 (gpt-oss): a API conta os parâmetros lógicos como U8, mas nos safetensors
    # eles vêm empacotados 2 por byte — 0.5 byte por parâmetro, não 1.
    if (config.get("quantization_config") or {}).get("quant_method") == "mxfp4":
        bytes_per_param = {**BYTES_PER_PARAM, "U8": 0.5}
    if parameters:
        weight_bytes = sum(int(count) * bytes_per_param.get(dtype, 2) for dtype, count in parameters.items())
    else:
        weight_bytes = int(safetensors.get("total", 0)) * 2
    if weight_bytes == 0:
        raise ValueError(f"{repo}: não achei o tamanho dos pesos (o repositório tem safetensors?)")
    try:
        heads = int(config["num_attention_heads"])
        info = ModelInfo(
            repo=repo,
            weight_bytes=weight_bytes,
            layers=int(config["num_hidden_layers"]),
            heads=heads,
            kv_heads=int(config.get("num_key_value_heads", heads)),
            head_dim=int(config.get("head_dim") or int(config["hidden_size"]) // heads),
            native_max_len=None,
        )
    except KeyError as error:
        raise ValueError(f"{repo}: config.json sem o campo {error}") from error
    native = config.get("max_position_embeddings")
    if native is not None:
        info = ModelInfo(**{**info.__dict__, "native_max_len": int(native)})
    return info


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


def _ctx_fit(info: ModelInfo, tensor_parallel: int, mem_gib: float, ceiling: float) -> int:
    """Maior contexto cujo KV cabe no teto de memória, com esse paralelismo."""
    room_gib = tensor_parallel * (mem_gib * ceiling - OVERHEAD_GIB) - weight_gib(info)
    if room_gib <= 0:
        return 0
    return int(room_gib * GIB / info.kv_bytes_per_token)


def _ceil_util(need_gib: float, mem_gib: float) -> float:
    """gpu-util derivado, arredondado para cima em 2 casas (margem mínima)."""
    return math.ceil(need_gib / mem_gib * 100) / 100


def solve(
    info: ModelInfo,
    partitions: list[Partition],
    *,
    max_len: int | None = None,
    gpu_util: float | None = None,
    gpus: int | None = None,
) -> tuple[list[Plan], list[Rejection]]:
    """Deriva planos (partição, GPUs, gpu-util, contexto) para o modelo.

    Sem overrides, o alvo é o contexto nativo do modelo (ou MIN_CONTEXT se o
    config.json não disser qual é) e o teto de gpu-util é GPU_UTIL_CEILING.
    Com overrides, eles viram o próprio alvo/teto.
    """
    ceiling = gpu_util if gpu_util is not None else GPU_UTIL_CEILING
    target = max_len or info.native_max_len or MIN_CONTEXT
    floor = max_len or MIN_CONTEXT
    plans: list[Plan] = []
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

        valid = [tp for tp in TP_CHOICES if tp <= part.max_node_gpus and info.heads % tp == 0]
        if gpus is not None:
            if gpus > part.max_node_gpus:
                rejections.append(
                    Rejection(part.name, f"tem no máximo {part.max_node_gpus} GPU(s) por nó; você pediu {gpus}")
                )
                continue
            if info.heads % gpus != 0:
                rejections.append(
                    Rejection(part.name, f"{gpus} GPU(s) não divide(m) as {info.heads} cabeças de atenção")
                )
                continue
            valid = [gpus]
        if not valid:
            rejections.append(
                Rejection(
                    part.name,
                    f"nenhum paralelismo de tensor com até {part.max_node_gpus} GPU(s) divide as {info.heads} cabeças de atenção",
                )
            )
            continue

        best: Plan | None = None
        for tp in valid:
            fit = _ctx_fit(info, tp, mem_gib, ceiling)
            if fit < floor:
                continue
            chosen = min(target, fit)
            need = required_gib(info, tp, chosen)
            plan = Plan(
                partition=part.name,
                gpu_type=part.gpu_type,
                gpus=tp,
                gpu_util=gpu_util if gpu_util is not None else _ceil_util(need, mem_gib),
                max_len=chosen,
                dtype=dtype_for(capability),
                need_gib=need,
                fits_now=part.max_node_free >= tp,
                free_now=part.max_node_free,
                pending=part.pending,
                full_context=chosen == target,
            )
            if plan.full_context:
                best = plan  # tp cresce na ordem de `valid`: o primeiro que serve tudo é o menor
                break
            if best is None or plan.max_len > best.max_len:
                best = plan
        if best is None:
            fit = _ctx_fit(info, valid[-1], mem_gib, ceiling)
            if max_len is not None:
                reason = (
                    f"com {valid[-1]} GPU(s) e gpu-util {ceiling} cabem {fit} tokens de contexto; "
                    f"você pediu {max_len}"
                )
            else:
                reason = (
                    f"nem o mínimo de {MIN_CONTEXT} tokens cabe com {valid[-1]} GPU(s) e "
                    f"gpu-util até {ceiling} (cabem {fit})"
                )
            rejections.append(Rejection(part.name, reason))
            continue
        plans.append(best)

    plans.sort(key=lambda p: (not p.full_context, not p.fits_now, p.gpus, p.gpu_util, p.partition))
    return plans, rejections

"""Padrões do kitsune. As opções de linha de comando sobrescrevem o que está aqui."""

from __future__ import annotations

import os
from pathlib import Path

HOST = os.environ.get("KITSUNE_HOST", "headnode")
REMOTE_ROOT = "kitsune"

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_CPUS = 4
DEFAULT_MEM = "16G"
DEFAULT_TIME = "01:00:00"

PYTHON_VERSION = "3.12"
# vLLM 0.11.0 pede torch 2.8.0 do PyPI, cujo wheel é cu128: roda no driver 570 do cluster (CUDA 12.8).
# gpt-oss entrou no vLLM 0.10; a 0.11.0 é a mais nova cujo torch ainda cabe nesse teto —
# as seguintes (0.26+, torch 2.11) usam CUDA 12.9/13 e falham com "driver too old".
VLLM_VERSION = "0.11.0"
TORCH_VERSION = "2.8.0"
# gpt-oss exige transformers >= 4.55; o vLLM 0.11.0 exige >= 4.55.2.
TRANSFORMERS_VERSION = "4.55.2"
# Teto de CUDA do driver 570: o setup falha cedo se o wheel do torch vier com CUDA maior.
MAX_CUDA_VERSION = 12.8

# Dimensionamento automático (ver DESIGN.md). O solucionador deriva o gpu-util do
# que o modelo precisa, limitado por este teto; 0.9 é o próprio padrão do vLLM e
# pressupõe GPU exclusiva via gres do Slurm.
GPU_UTIL_CEILING = 0.9
# Contexto mínimo aceitável num plano automático; abaixo disso, a partição é recusada.
MIN_CONTEXT = 4096

# Tipo de GPU (como aparece no Gres do Slurm) -> (memória em GiB, compute capability).
# Fontes, por tipo:
#   L40S: nvidia-smi no manual do cluster (46068 MiB); cc 8.9 na tabela da NVIDIA.
#   RTX8000, RTX5000, P5000: sondados com nvidia-smi nos nós (jobs de 3 min, 2026-10-27).
#   RTXA5000: 24 GB na página da NVIDIA; cc 8.6 na tabela da NVIDIA.
#   RTXPRO6000: 96 GB na página da NVIDIA (95 GiB é a leitura do nvidia-smi); cc 12.0 na tabela.
#   H100: cc 9.0 na tabela da NVIDIA; memória NÃO verificada (dl-04 está fora do ar).
GPU_SPECS = {
    "L40S": (45, 8.9),
    "H100": (93, 9.0),
    "RTXA5000": (24, 8.6),
    "RTX8000": (48, 7.5),
    "RTX5000": (16, 7.5),
    "P5000": (16, 6.1),
    "RTXPRO6000": (95, 12.0),
}
# A documentação do vLLM diz "compute capability 7.5 or higher".
MIN_COMPUTE_CAPABILITY = 7.5

API_KEY_FILE = Path.home() / ".local" / "state" / "kitsune" / "api_key"
TUNNEL_PORT = 8000
# Pasta dos chats, relativa ao diretório de onde o comando roda.
LEDGER_DIR = "ledger-kitsune"

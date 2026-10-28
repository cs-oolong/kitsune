"""Script de batch que sobe o vLLM num nó de GPU.

O script vai pela entrada padrão do sbatch, então nada precisa ser copiado para o
headnode antes do envio. O token é gerado no próprio nó e fica só no arquivo de
estado, com permissão 600; nunca aparece na linha de comando.
"""

from __future__ import annotations

import shlex

from .config import REMOTE_ROOT
from .remote import model_dir_name

_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=kitsune
#SBATCH --partition={partition}
#SBATCH --gres=gpu:{gpus}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --output=logs/%j.out

set -euo pipefail
ROOT="$HOME/{root}"
VENV="$ROOT/venv"
export HF_HUB_OFFLINE=1
MODEL_DIR="$ROOT/models/{name}"
STATE="$ROOT/state/$SLURM_JOB_ID.env"
umask 077
mkdir -p "$ROOT/state"

PORT=$("$VENV/bin/python" -c "import socket; s = socket.socket(); s.bind(('', 0)); print(s.getsockname()[1])")
TOKEN=$("$VENV/bin/python" -c "import secrets; print(secrets.token_hex(24))")

write_state() {{
    printf 'JOBID=%s\\nNODE=%s\\nPORT=%s\\nMODEL=%s\\nTOKEN=%s\\nSTATUS=%s\\n' \\
        "$SLURM_JOB_ID" "$(hostname)" "$PORT" {name_quoted} "$TOKEN" "$1" > "$STATE.tmp"
    mv "$STATE.tmp" "$STATE"
}}

write_state loading
export VLLM_API_KEY="$TOKEN"
"$VENV/bin/vllm" serve "$MODEL_DIR" --served-model-name {name_quoted} --host 0.0.0.0 --port "$PORT" \\
    --tensor-parallel-size {gpus} --dtype {dtype} --max-model-len {max_len} \\
    --gpu-memory-utilization {gpu_util}{trust_flag} &
SERVER=$!

until "$VENV/bin/python" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:$PORT/health', timeout=2)" 2>/dev/null; do
    if ! kill -0 "$SERVER" 2>/dev/null; then
        write_state failed
        exit 1
    fi
    sleep 5
done

write_state ready
echo "kitsune pronto em $(hostname):$PORT"
wait "$SERVER"
"""


def render_job_script(
    model: str,
    *,
    partition: str,
    gpus: int,
    cpus: int,
    mem: str,
    time: str,
    dtype: str,
    gpu_util: float,
    max_len: int,
    trust_remote_code: bool = False,
) -> str:
    name = model_dir_name(model)
    trust_flag = " --trust-remote-code" if trust_remote_code else ""
    return _TEMPLATE.format(
        partition=partition,
        gpus=gpus,
        cpus=cpus,
        mem=mem,
        time=time,
        root=REMOTE_ROOT,
        name=name,
        name_quoted=shlex.quote(name),
        dtype=dtype,
        max_len=max_len,
        gpu_util=gpu_util,
        trust_flag=trust_flag,
    )

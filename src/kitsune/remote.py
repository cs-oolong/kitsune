"""Tudo que o kitsune faz no cluster passa por `ssh headnode`. Nunca roda no login node."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass

from .config import (
    HOST,
    MAX_CUDA_VERSION,
    PYTHON_VERSION,
    REMOTE_ROOT,
    TORCH_VERSION,
    TRANSFORMERS_VERSION,
    VLLM_VERSION,
)

JOB_NAME = "kitsune"
SQUEUE_FORMAT = "%i|%T|%R|%M"
MODEL_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")
REPO_MARKER = ".kitsune-repo"
MODELS_SCRIPT = """for dir in "$HOME/kitsune/models"/*/; do
    [ -d "$dir" ] || continue
    name=$(basename "$dir")
    repo=$(head -n 1 "$dir/.kitsune-repo" 2>/dev/null || true)
    size=$(du -sh "$dir" 2>/dev/null | cut -f1)
    if [ -f "$dir/config.json" ]; then complete=sim; else complete=nao; fi
    echo "$name|${repo:-?}|$size|$complete"
done
"""


class RemoteError(RuntimeError):
    pass


@dataclass(frozen=True)
class Job:
    job_id: str
    state: str
    where: str
    elapsed: str


@dataclass(frozen=True)
class ModelEntry:
    folder: str
    repo: str
    size: str
    complete: bool


def validate_model(model: str) -> str:
    if not MODEL_PATTERN.fullmatch(model):
        raise ValueError(f"nome de modelo inválido: {model!r} (use autor/repo do Hugging Face)")
    return model


def model_dir_name(model: str) -> str:
    return model.split("/", 1)[1]


def run(command: str, *, stdin_text: str | None = None, capture: bool = True, check: bool = True):
    try:
        result = subprocess.run(
            ["ssh", HOST, command],
            input=stdin_text,
            text=True,
            capture_output=capture,
        )
    except FileNotFoundError as error:
        raise RemoteError("ssh não encontrado no PATH") from error
    if check and result.returncode != 0:
        detail = (result.stderr or "").strip() if capture else ""
        message = f"ssh {HOST} falhou (código {result.returncode})"
        raise RemoteError(f"{message}: {detail}" if detail else message)
    return result


def parse_squeue(text: str) -> list[Job]:
    jobs = []
    for line in text.splitlines():
        parts = line.strip().split("|")
        if len(parts) == 4 and parts[0]:
            jobs.append(Job(*parts))
    return jobs


def parse_state(text: str) -> dict[str, str]:
    values = {}
    for line in text.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def parse_models(text: str) -> list[ModelEntry]:
    entries = []
    for line in text.splitlines():
        parts = line.strip().split("|")
        if len(parts) == 4 and parts[0]:
            entries.append(ModelEntry(parts[0], parts[1], parts[2], parts[3] == "sim"))
    return sorted(entries, key=lambda entry: entry.folder)


def list_models() -> list[ModelEntry]:
    return parse_models(run("bash -s", stdin_text=MODELS_SCRIPT).stdout)


def list_jobs() -> list[Job]:
    result = run(f'squeue -h -u "$USER" -n {JOB_NAME} -o "{SQUEUE_FORMAT}"')
    return parse_squeue(result.stdout)


def read_state(job_id: str) -> dict[str, str]:
    _check_job_id(job_id)
    result = run(f'cat "$HOME/{REMOTE_ROOT}/state/{job_id}.env"', capture=True, check=False)
    return parse_state(result.stdout) if result.returncode == 0 else {}


def check_setup(model: str) -> None:
    name = model_dir_name(model)
    command = (
        f'test -x "$HOME/{REMOTE_ROOT}/venv/bin/vllm" '
        f'&& test -f "$HOME/{REMOTE_ROOT}/models/{name}/config.json"'
    )
    if run(command, check=False).returncode != 0:
        raise RemoteError(f"ambiente ou modelo {model} não está pronto no cluster. Rode `kitsune setup` primeiro.")


def submit(script: str) -> str:
    result = run(f'cd "$HOME/{REMOTE_ROOT}" && mkdir -p logs && sbatch --parsable', stdin_text=script)
    return result.stdout.strip().split(";")[0]


def cancel(job_id: str) -> None:
    _check_job_id(job_id)
    run(f"scancel {job_id}")


def _check_job_id(job_id: str) -> None:
    if not job_id.isdigit():
        raise ValueError(f"id de job inválido: {job_id!r}")


def setup_script(model: str) -> str:
    name = model_dir_name(model)
    return f"""set -euo pipefail
MODEL='{model}'
ROOT="$HOME/{REMOTE_ROOT}"
MODEL_DIR="$ROOT/models/{name}"
mkdir -p "$ROOT/logs" "$ROOT/state" "$ROOT/models"
if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
UV=$(command -v uv 2>/dev/null || echo "$HOME/.local/bin/uv")
# Em home NFS, rm -rf falha em arquivos .nfs* de processos que ainda seguram o venv;
# renomear funciona sempre e o backup sai depois, quando os donos fecharem os arquivos.
if [ -x "$ROOT/venv/bin/python" ] && ! "$ROOT/venv/bin/python" -c "import vllm; assert vllm.__version__ == '{VLLM_VERSION}'" >/dev/null 2>&1; then
    mv "$ROOT/venv" "$ROOT/venv.bak.$(date +%s)"
fi
if [ ! -x "$ROOT/venv/bin/python" ]; then
    "$UV" venv --python {PYTHON_VERSION} "$ROOT/venv"
fi
"$UV" pip install --python "$ROOT/venv/bin/python" vllm=={VLLM_VERSION} transformers=={TRANSFORMERS_VERSION} huggingface_hub
"$ROOT/venv/bin/python" -c "import torch; assert torch.__version__.startswith('{TORCH_VERSION}') and float(torch.version.cuda) <= {MAX_CUDA_VERSION}, (torch.__version__, torch.version.cuda)"
# gpt-oss: o tokenizer Harmony baixa um vocab de openaipublic na 1ª vez e os nós de
# computação podem não ter rota. Pré-baixa no headnode, num cache que o job exporta.
export TIKTOKEN_RS_CACHE_DIR="$ROOT/tiktoken-cache"
mkdir -p "$TIKTOKEN_RS_CACHE_DIR"
if ! "$ROOT/venv/bin/python" -c "from openai_harmony import load_harmony_encoding; load_harmony_encoding('HarmonyGptOss')" 2>/dev/null; then
    case "$MODEL" in
        *gpt-oss*) echo "erro: não consegui baixar o vocab Harmony (openaipublic); sem ele o modelo não sobe" >&2; exit 1;;
        *) echo "aviso: pré-download do vocab Harmony falhou; só afeta modelos gpt-oss" >&2;;
    esac
fi
"$ROOT/venv/bin/python" -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='$MODEL', local_dir='$MODEL_DIR')"
echo "$MODEL" > "$MODEL_DIR/{REPO_MARKER}"
echo "Modelo pronto em $MODEL_DIR"
"""

"""Ponto de entrada do kitsune."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sys
import tempfile
import time
from pathlib import Path
from typing import Sequence

from . import __version__
from . import config
from .chatlog.errors import ChatError
from .chatlog.ledger import history as ledger_history, latest_chat, load_chat, read_chats, validate_chat_id
from .chatlog.turn import run_turn
from .client import save_api_key, tunnel
from .cluster import Inventory, fetch_inventory
from .estimate import OVERHEAD_GIB, ModelInfo, Plan, dtype_for, fetch_model_info, kv_gib, solve, weight_gib
from .job import render_job_script
from .remote import (
    RemoteError,
    cancel,
    check_setup,
    last_job,
    list_jobs,
    list_models,
    read_log,
    read_state,
    run,
    setup_script,
    submit,
    validate_model,
)

PROVIDER = "kitsune"
STATUS_LABEL = {
    "loading": "carregando o modelo",
    "ready": "pronto",
    "failed": "falhou",
}
PREVIEW_CHARS = 60
CAUSE_LOG_LINES = 300
CAUSE_CHARS = 300
END_REASON = {
    "CANCELLED": "foi cancelado",
    "TIMEOUT": "estourou o limite de tempo",
    "COMPLETED": "encerrou",
    "PREEMPTED": "foi preemptado",
}
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_PID_PREFIX = re.compile(r"^\([A-Za-z_]+ pid=\d+\)\s*")
_TRACEBACK = "Traceback (most recent call last)"
_ERRORISH = re.compile(r"\berror\b|\bfailed\b|\bexception\b", re.IGNORECASE)


def extract_cause(log_text: str) -> str | None:
    """Melhor palpite sobre a causa da falha, a partir do fim do log do job.

    Prefere a linha da exceção do último traceback; sem traceback, a última
    linha com cara de erro. Tira códigos ANSI e o prefixo "(Worker pid=N)" do vLLM.
    """
    lines = [
        _PID_PREFIX.sub("", _ANSI.sub("", raw)).strip()
        for raw in log_text.replace("\r", "\n").splitlines()
    ]
    lines = [line for line in lines if line]
    start = max((i for i, line in enumerate(lines) if _TRACEBACK in line), default=None)
    if start is not None:
        cause = None
        skip_next = False  # linha de código-fonte logo abaixo de uma moldura "File ..."
        for line in lines[start + 1 :]:
            if line.startswith("File "):
                skip_next = True
                continue
            if line.startswith(("^", "~")):
                continue
            if skip_next:
                skip_next = False
                continue
            cause = line
        if cause is not None:
            return cause[:CAUSE_CHARS]
    for line in reversed(lines):
        if _ERRORISH.search(line):
            return line[:CAUSE_CHARS]
    return None


def cmd_setup(args: argparse.Namespace) -> int:
    model = validate_model(args.model)
    print(f"Preparando o headnode: uv, ambiente com vLLM e o modelo {model}. Pode demorar.")
    run("bash -s", stdin_text=setup_script(model), capture=False)
    print("Pronto. Próximo passo: kitsune suggest, depois kitsune start")
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    models = list_models()
    if not models:
        print("Nenhum modelo baixado. Use `kitsune setup --model autor/repo`.")
        return 0
    rows = [("MODELO", "PASTA", "TAMANHO", "STATUS")]
    for model in models:
        status = "completo" if model.complete else "incompleto (sem config.json)"
        rows.append((model.repo, model.folder, model.size, status))
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    for row in rows:
        print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip())
    if any(model.repo == "?" for model in models):
        print("\n? = baixado antes do marcador; rode `kitsune setup` de novo para gravar o nome do repositório.")
    return 0


def cmd_start(args: argparse.Namespace) -> int:
    model = validate_model(args.model)
    active = list_jobs()
    if active:
        job = active[0]
        raise RemoteError(
            f"já existe um kitsune no ar (job {job.job_id}, {job.state}). Use `kitsune stop` antes."
        )
    inventory = fetch_inventory()
    manual = (args.partition, args.gpus, args.gpu_util, args.max_len)
    if all(value is not None for value in manual):
        # Caminho manual: as quatro flags dispensam a consulta ao Hugging Face.
        partition = inventory.partition(args.partition)
        if partition is None:
            raise RemoteError(f"a partição {args.partition} não existe. Veja `kitsune cluster`.")
        if partition.max_node_gpus == 0:
            raise RemoteError(f"a partição {args.partition} não tem nenhum nó utilizável agora.")
        if args.gpus > partition.max_node_gpus:
            raise RemoteError(
                f"a partição {args.partition} tem no máximo {partition.max_node_gpus} GPU(s) por nó; "
                f"você pediu {args.gpus}."
            )
        capability = config.GPU_SPECS.get(partition.gpu_type or "", (0, 0.0))[1]
        plan_kwargs = {
            "partition": partition.name,
            "gpus": args.gpus,
            "gpu_util": args.gpu_util,
            "max_len": args.max_len,
            "dtype": dtype_for(capability) if capability else "auto",
        }
        fits_now = partition.max_node_free >= args.gpus
        print(
            f"Configuração manual: {partition.name}, {args.gpus} GPU(s), gpu-util {args.gpu_util}, "
            f"contexto {args.max_len}, dtype {plan_kwargs['dtype']}."
        )
    else:
        partitions = inventory.partitions()
        if args.partition is not None:
            only = inventory.partition(args.partition)
            if only is None:
                raise RemoteError(f"a partição {args.partition} não existe. Veja `kitsune cluster`.")
            partitions = [only]
        try:
            info = fetch_model_info(model)
        except ValueError as error:
            raise RemoteError(
                f"{error} Sem a consulta ao Hugging Face não dá para calcular o plano; "
                "passe --partition, --gpus, --gpu-util e --max-len explicitamente."
            ) from error
        plans, rejected = solve(info, partitions, max_len=args.max_len, gpu_util=args.gpu_util, gpus=args.gpus)
        if not plans:
            why = "; ".join(f"{item.partition}: {item.reason}" for item in rejected) or "nenhuma partição com GPU"
            raise RemoteError(f"nenhuma configuração serve para {model}: {why}")
        plan = plans[0]
        plan_kwargs = {
            "partition": plan.partition,
            "gpus": plan.gpus,
            "gpu_util": plan.gpu_util,
            "max_len": plan.max_len,
            "dtype": plan.dtype,
        }
        fits_now = plan.fits_now
        print("Plano: " + _plan_line(plan, info))

    check_setup(model)
    script = render_job_script(
        model,
        cpus=config.DEFAULT_CPUS,
        mem=config.DEFAULT_MEM,
        time=args.time,
        trust_remote_code=args.trust_remote_code,
        **plan_kwargs,
    )
    job_id = submit(script)
    print(f"Job {job_id} enviado: {model} em {plan_kwargs['partition']}, {plan_kwargs['gpus']} GPU, limite {args.time}.")
    if not fits_now:
        print("Aviso: agora não há GPUs suficientes livres num nó. O job vai esperar na fila.")
    print("Acompanhe com: kitsune status")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    jobs = list_jobs()
    if not jobs:
        return _status_last_job()
    for job in jobs:
        if job.state == "RUNNING":
            state = read_state(job.job_id)
            label = STATUS_LABEL.get(state.get("STATUS", ""), "iniciando")
            if state.get("STATUS") == "failed":
                label += f" — veja com `kitsune logs {job.job_id}`"
            node = state.get("NODE", job.where)
            print(f"job {job.job_id}: {label} em {node} (rodando há {job.elapsed})")
        else:
            print(f"job {job.job_id}: {job.state.lower()} ({job.where}), tempo {job.elapsed}")
    return 0


def _status_last_job() -> int:
    report = last_job()
    if report is None:
        print("Nenhum servidor no ar. Use `kitsune start`.")
        return 0
    slurm = (report.slurm_state or "").split()[0] if report.slurm_state else ""
    if slurm == "FAILED" or report.state.get("STATUS") == "failed":
        suffix = f" (saída {report.exit_code})" if report.exit_code else ""
        print(f"Nenhum servidor no ar. O último job ({report.job_id}) falhou{suffix}.")
        try:
            _, log = read_log(report.job_id, lines=CAUSE_LOG_LINES)
            cause = extract_cause(log)
        except RemoteError:
            cause = None
        if cause:
            print(f"Causa provável: {cause}")
        print(f"Veja o log com: kitsune logs {report.job_id}")
        return 0
    reason = END_REASON.get(slurm)
    if reason is None:
        reason = f"encerrou ({slurm.lower()})" if slurm else "encerrou (parado ou tempo esgotado)"
    print(f"Nenhum servidor no ar. O último job ({report.job_id}) {reason}. Use `kitsune start` para subir outro.")
    return 0


def cmd_logs(args: argparse.Namespace) -> int:
    job_id, text = read_log(args.job_id, lines=args.lines)
    header = f"== ~/kitsune/logs/{job_id}.out"
    if args.lines > 0:
        header += f" (últimas {args.lines} linhas)"
    print(header + " ==")
    sys.stdout.write(text if text.endswith("\n") or not text else text + "\n")
    return 0


def cmd_cluster(args: argparse.Namespace) -> int:
    inventory = fetch_inventory()
    print(render_cluster(inventory))
    return 0


def render_cluster(inventory: Inventory) -> str:
    rows = [("PARTIÇÃO", "GPU", "LIVRES/TOTAL", "MAIOR NÓ LIVRE", "NÓS", "FORA DO AR", "FILA")]
    for part in inventory.partitions():
        if part.gpu_type is None:
            continue
        rows.append(
            (
                part.name + ("*" if part.is_default else ""),
                part.gpu_type,
                f"{part.gpus_free}/{part.gpus_total}",
                str(part.max_node_free),
                str(part.nodes),
                str(part.nodes_down),
                str(part.pending),
            )
        )
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    lines = ["  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip() for row in rows]
    lines.append("")
    lines.append("* partição padrão. FILA conta os jobs pendentes que aceitam a partição.")
    lines.append("LIVRES/TOTAL considera só os nós que estão no ar.")
    return "\n".join(lines)


def cmd_suggest(args: argparse.Namespace) -> int:
    model = validate_model(args.model)
    info = fetch_model_info(model)
    partitions = fetch_inventory().partitions()
    if args.partition is not None:
        only = next((part for part in partitions if part.name == args.partition), None)
        if only is None:
            raise RemoteError(f"a partição {args.partition} não existe. Veja `kitsune cluster`.")
        partitions = [only]
    plans, rejected = solve(info, partitions, max_len=args.max_len, gpu_util=args.gpu_util, gpus=args.gpus)

    ceiling = args.gpu_util if args.gpu_util is not None else config.GPU_UTIL_CEILING
    target = args.max_len or info.native_max_len or config.MIN_CONTEXT
    native = str(info.native_max_len) if info.native_max_len else f"desconhecido; usando o piso {config.MIN_CONTEXT}"
    print(f"Modelo {model}")
    print(
        f"  pesos {weight_gib(info):.2f} GiB; KV cache para {target} tokens {kv_gib(info, target):.2f} GiB; "
        f"overhead {OVERHEAD_GIB:.0f} GiB por GPU"
    )
    print(f"  contexto nativo {native}; teto de gpu-util {ceiling}\n")

    if plans:
        best = plans[0]
        print("Recomendado:")
        print(f"  {_start_command(model, args)}")
        print("  " + _plan_line(best, info))
        if not best.fits_now:
            print(f"  atenção: agora não há {best.gpus} GPU(s) livres num nó; vai esperar na fila")
        if len(plans) > 1:
            print("\nOutras opções:")
            for option in plans[1:]:
                print(
                    f"  {option.partition} ({option.gpu_type}): {option.gpus} GPU(s), gpu-util {option.gpu_util}, "
                    f"contexto {option.max_len}, livres agora {option.free_now}, fila {option.pending}"
                )
    else:
        print("Nenhuma partição serve com essa configuração.")
        if args.gpu_util is not None:
            wider, _ = solve(info, partitions, max_len=args.max_len, gpus=args.gpus)
            if wider:
                names = ", ".join(f"{plan.partition} ({plan.gpus} GPU)" for plan in wider)
                print(f"Com gpu-util automático (teto {config.GPU_UTIL_CEILING}) caberia em: {names}.")

    if rejected:
        print("\nDescartadas:")
        for item in rejected:
            print(f"  {item.partition}: {item.reason}")
    return 0


def _start_command(model: str, args: argparse.Namespace) -> str:
    parts = ["kitsune start", f"--model {model}"]
    for flag in ("partition", "gpus", "gpu_util", "max_len"):
        value = getattr(args, flag, None)
        if value is not None:
            parts.append(f"--{flag.replace('_', '-')} {value}")
    return " ".join(parts)


def _plan_line(plan: Plan, info: ModelInfo) -> str:
    context = f"contexto {plan.max_len}"
    if info.native_max_len:
        context += " (nativo)" if plan.full_context else f" (limitado; o nativo é {info.native_max_len})"
    return (
        f"{plan.partition} ({plan.gpu_type}), {plan.gpus} GPU(s), dtype {plan.dtype}, "
        f"gpu-util {plan.gpu_util}, {context}; precisa {plan.need_gib:.1f} GiB por GPU"
    )


def check_chat_target(ledger_dir: Path, chat_id: str | None, system: str | None) -> None:
    if chat_id is None:
        return
    validate_chat_id(chat_id)
    if system is not None:
        raise ChatError("--system só vale para chat novo; não use junto com -c", code="bad_request")
    if load_chat(ledger_dir, chat_id) is None:
        raise ChatError(
            f"chat {chat_id} não existe em {ledger_dir}. Sem -c, o comando cria um chat novo.",
            code="chat_not_found",
        )


def _ready_server() -> tuple[str, dict[str, str]]:
    running = [job for job in list_jobs() if job.state == "RUNNING"]
    if not running:
        raise RemoteError("nenhum servidor no ar. Use `kitsune start` e acompanhe com `kitsune status`.")
    job_id = running[0].job_id
    state = read_state(job_id)
    if state.get("STATUS") != "ready":
        label = STATUS_LABEL.get(state.get("STATUS", ""), "iniciando")
        raise RemoteError(f"o servidor ainda não está pronto ({label}). Veja `kitsune status`.")
    return job_id, state


def cmd_chat(args: argparse.Namespace) -> int:
    text = " ".join(args.message).strip()
    if not text:
        raise ValueError("escreva uma mensagem")
    ledger_dir = Path(args.ledger_dir)
    check_chat_target(ledger_dir, args.chat, args.system)

    _, state = _ready_server()

    system_file = Path(args.system_prompt_file)
    temp_path = None
    if args.system is not None:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", encoding="utf-8", delete=False) as handle:
            handle.write(args.system)
            temp_path = Path(handle.name)
        system_file = temp_path
    os.environ["KITSUNE_API_KEY"] = state["TOKEN"]
    try:
        with tunnel(state["NODE"], int(state["PORT"])) as local_port:
            result = run_turn(
                state["MODEL"],
                text,
                chat_id=args.chat,
                ledger_dir=ledger_dir,
                system_prompt_file=system_file,
                provider_name=PROVIDER,
                base_url=f"http://127.0.0.1:{local_port}/v1",
            )
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    last = result["history"][-1]
    if "error" in last:
        print(f"erro: {last['error']}", file=sys.stderr)
        print(f"[chat {result['chat_id']}]", file=sys.stderr)
        return 1
    print(last["content"])
    print(f"\n[chat {result['chat_id']}]", file=sys.stderr)
    return 0


def _raise_interrupt(*_: object) -> None:
    raise KeyboardInterrupt()


def cmd_tunnel(args: argparse.Namespace) -> int:
    sys.stdout.reconfigure(line_buffering=True)
    # Processos de fundo de shells não interativos herdam SIGINT ignorado; SIGTERM também fecha o túnel.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, _raise_interrupt)
    job_id, state = _ready_server()

    save_api_key(state["TOKEN"], config.API_KEY_FILE)
    with tunnel(state["NODE"], int(state["PORT"]), local_port=args.port) as local_port:
        print(f"Túnel aberto: http://127.0.0.1:{local_port}/v1 -> {state['NODE']}:{state['PORT']} (job {job_id})")
        print(f"Modelo: {state['MODEL']}")
        print(f"Chave: {config.API_KEY_FILE} (permissão 600)")
        print("\nEm outro terminal:")
        print(f'  export KITSUNE_API_KEY="$(cat {config.API_KEY_FILE})"')
        body = json.dumps({"model": state["MODEL"], "messages": [{"role": "user", "content": "oi"}]}, ensure_ascii=False)
        print(
            f"  curl http://127.0.0.1:{local_port}/v1/chat/completions "
            f'-H "Authorization: Bearer $KITSUNE_API_KEY" -H "Content-Type: application/json" -d \'{body}\''
        )
        print("\nCtrl-C fecha o túnel.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nTúnel fechado.")
    return 0


def cmd_chats(args: argparse.Namespace) -> int:
    chats = read_chats(Path(args.ledger_dir))
    if not chats:
        print('Nenhum chat ainda. Use `kitsune chat "..."`.')
        return 0
    print("ID                              ATUALIZADO                PROVEDOR  MODELO                       TURNOS  PRIMEIRA MENSAGEM")
    for chat in chats:
        messages = chat.get("messages", [])
        turns = sum(1 for m in messages if m.get("role") == "user")
        first = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
        preview = " ".join(first.split())[:PREVIEW_CHARS] or "(vazio)"
        print(f"{chat['chat_id']:<31} {chat.get('updated_at', ''):<25} {chat.get('provider', ''):<9} {chat.get('model_id', ''):<28} {turns:>6}  {preview}")
    return 0


def cmd_history(args: argparse.Namespace) -> int:
    ledger_dir = Path(args.ledger_dir)
    chat_id = args.chat_id or latest_chat(ledger_dir, provider=PROVIDER)
    if chat_id is None:
        print('Nenhum chat do kitsune. Use `kitsune chat "..."` ou `kitsune chats`.')
        return 0
    chat = load_chat(ledger_dir, chat_id)
    if chat is None:
        raise ValueError(f"chat {chat_id} não existe em {ledger_dir}")
    print(f"Chat {chat_id} ({chat.get('provider')}/{chat.get('model_id')}, criado em {chat.get('created_at')})\n")
    print(f"[sistema] {chat.get('system_prompt', '')}\n")
    for message in ledger_history(chat, include_errors=True):
        if "error" in message:
            print(f"Modelo: (erro) {message['error']}\n")
            continue
        who = "Você" if message["role"] == "user" else "Modelo"
        print(f"{who}: {message['content']}\n")
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    jobs = list_jobs()
    if not jobs:
        print("Nada para parar.")
        return 0
    for job in jobs:
        cancel(job.job_id)
        print(f"job {job.job_id} cancelado.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kitsune",
        description="Sobe um modelo num nó de GPU do cluster por tempo limitado e conversa com ele.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    setup = sub.add_parser("setup", help="prepara o headnode: uv, vLLM e o modelo")
    setup.add_argument("--model", default=config.DEFAULT_MODEL, help="autor/repo do Hugging Face")
    setup.set_defaults(func=cmd_setup)

    models = sub.add_parser("models", help="lista os modelos baixados no headnode")
    models.set_defaults(func=cmd_models)

    cluster = sub.add_parser("cluster", help="mostra GPUs livres por partição e a fila")
    cluster.set_defaults(func=cmd_cluster)

    suggest_cmd = sub.add_parser("suggest", help="mostra o plano (partição, GPUs, gpu-util, contexto) sem enviar")
    suggest_cmd.add_argument("--model", default=config.DEFAULT_MODEL, help="autor/repo do Hugging Face")
    suggest_cmd.add_argument("--partition", default=None, help="e se eu usar esta partição? (padrão: todas)")
    suggest_cmd.add_argument("--gpus", type=int, default=None, help="e se eu pedir este número de GPUs? (padrão: calculado)")
    suggest_cmd.add_argument("--gpu-util", type=float, default=None, help="e se o teto de gpu-util for este? (padrão: %(default)s = automático)")
    suggest_cmd.add_argument("--max-len", type=int, default=None, help="e se o contexto for este? (padrão: o nativo do modelo)")
    suggest_cmd.set_defaults(func=cmd_suggest)

    start = sub.add_parser("start", help="calcula o plano e sobe o modelo num nó de GPU")
    start.add_argument("--model", default=config.DEFAULT_MODEL, help="autor/repo do Hugging Face")
    start.add_argument("--partition", default=None, help="override; sem flag, o solucionador escolhe")
    start.add_argument("--gpus", type=int, default=None, help="override; sem flag, o solucionador escolhe")
    start.add_argument("--time", default=config.DEFAULT_TIME, help="limite do job, HH:MM:SS")
    start.add_argument("--gpu-util", type=float, default=None, help="override; sem flag, derivado do que o modelo precisa")
    start.add_argument("--max-len", type=int, default=None, help="override; sem flag, o contexto nativo do modelo (limitado ao que cabe)")
    start.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="deixa o vLLM executar o código Python que vem no repositório do modelo",
    )
    start.set_defaults(func=cmd_start)

    status = sub.add_parser("status", help="mostra se há um servidor no ar; sem job vivo, o que aconteceu com o último")
    status.set_defaults(func=cmd_status)

    logs = sub.add_parser("logs", help="mostra o log de um job (o mais recente, se não passar o id)")
    logs.add_argument("job_id", nargs="?", metavar="JOBID", help="id do job no Slurm (padrão: o mais recente)")
    logs.add_argument("--lines", type=int, default=CAUSE_LOG_LINES // 6, help="quantas linhas do fim mostrar; 0 = o log inteiro (padrão: %(default)s)")
    logs.set_defaults(func=cmd_logs)

    chat = sub.add_parser("chat", help="manda uma mensagem ao modelo e grava em ledger-kitsune/")
    chat.add_argument("message", nargs="+")
    chat.add_argument("-c", "--chat", metavar="ID", help="continua o chat ID (precisa existir); sem -c, cria um chat novo")
    chat.add_argument("--system", default=None, help="instrução de sistema para um chat novo (dispensa o arquivo)")
    chat.add_argument("--system-prompt-file", default="system_prompt.txt", help="arquivo do prompt de sistema (padrão: %(default)s)")
    chat.add_argument("--ledger-dir", default=config.LEDGER_DIR, help="pasta do ledger (padrão: %(default)s)")
    chat.set_defaults(func=cmd_chat)

    tunnel_cmd = sub.add_parser("tunnel", help="abre o túnel até o servidor, para outros programas usarem a API")
    tunnel_cmd.add_argument("--port", type=int, default=config.TUNNEL_PORT, help="porta local (padrão: %(default)s)")
    tunnel_cmd.set_defaults(func=cmd_tunnel)

    chats = sub.add_parser("chats", help="lista os chats no ledger")
    chats.add_argument("--ledger-dir", default=config.LEDGER_DIR, help="pasta do ledger (padrão: %(default)s)")
    chats.set_defaults(func=cmd_chats)

    history = sub.add_parser("history", help="mostra um chat do ledger (o mais recente do kitsune, se não passar ID)")
    history.add_argument("chat_id", nargs="?", metavar="ID")
    history.add_argument("--ledger-dir", default=config.LEDGER_DIR, help="pasta do ledger (padrão: %(default)s)")
    history.set_defaults(func=cmd_history)

    stop = sub.add_parser("stop", help="cancela o servidor no ar")
    stop.set_defaults(func=cmd_stop)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ChatError as error:
        print(json.dumps({"error": error.code, "detail": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2
    except (RemoteError, ValueError) as error:
        print(f"erro: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130

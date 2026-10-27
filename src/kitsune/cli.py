"""Ponto de entrada do kitsune."""

from __future__ import annotations

import argparse
import signal
import sys
import time
from typing import Sequence

from . import __version__
from . import config
from .client import ChatError, chat_completion, save_api_key, tunnel
from .cluster import Inventory, fetch_inventory
from .conversation import ChatStore
from .estimate import OVERHEAD_GIB, dtype_for, fetch_model_info, kv_gib, suggest, weight_gib
from .job import render_job_script
from .remote import (
    RemoteError,
    cancel,
    check_setup,
    list_jobs,
    list_models,
    read_state,
    run,
    setup_script,
    submit,
    validate_model,
)

STATUS_LABEL = {
    "loading": "carregando o modelo",
    "ready": "pronto",
    "failed": "falhou (veja os logs em ~/kitsune/logs)",
}
PREVIEW_CHARS = 60
SUGGEST_WIDER_UTIL = 0.9


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
    dtype = dtype_for(capability) if capability else "auto"

    check_setup(model)
    script = render_job_script(
        model,
        partition=args.partition,
        gpus=args.gpus,
        cpus=args.cpus,
        mem=args.mem,
        time=args.time,
        dtype=dtype,
        gpu_util=args.gpu_util,
        max_len=args.max_len,
    )
    job_id = submit(script)
    print(f"Job {job_id} enviado: {model} em {args.partition}, {args.gpus} GPU, limite {args.time}.")
    if partition.max_node_free < args.gpus:
        print(
            f"Aviso: agora não há {args.gpus} GPU(s) livres num nó da {args.partition} "
            f"(maior nó livre: {partition.max_node_free}). O job vai esperar na fila."
        )
    print("Acompanhe com: kitsune status")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    jobs = list_jobs()
    if not jobs:
        print("Nenhum servidor no ar. Use `kitsune start`.")
        return 0
    for job in jobs:
        if job.state == "RUNNING":
            state = read_state(job.job_id)
            label = STATUS_LABEL.get(state.get("STATUS", ""), "iniciando")
            node = state.get("NODE", job.where)
            print(f"job {job.job_id}: {label} em {node} (rodando há {job.elapsed})")
        else:
            print(f"job {job.job_id}: {job.state.lower()} ({job.where}), tempo {job.elapsed}")
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
    suggestions, rejected = suggest(info, partitions, gpu_util=args.gpu_util, max_len=args.max_len)

    print(f"Modelo {model}")
    print(
        f"  pesos {weight_gib(info):.2f} GiB; KV cache para {args.max_len} tokens "
        f"{kv_gib(info, args.max_len):.2f} GiB; overhead {OVERHEAD_GIB:.0f} GiB por GPU"
    )
    print(f"  cada GPU usa gpu-util {args.gpu_util} da memória dela\n")

    if suggestions:
        best = suggestions[0]
        print("Recomendado:")
        print(f"  {_start_command(model, best.partition, best.gpus, args.gpu_util, args.max_len)}")
        print(
            f"  {best.partition} ({best.gpu_type}), {best.gpus} GPU(s), dtype {best.dtype}; "
            f"precisa {best.need_gib:.1f} GiB por GPU, tem {best.usable_gib:.1f} GiB"
        )
        if best.free_now < best.gpus:
            print(f"  atenção: agora não há {best.gpus} GPU(s) livres num nó; vai esperar na fila")
        if len(suggestions) > 1:
            print("\nOutras opções:")
            for option in suggestions[1:]:
                print(
                    f"  {option.partition} ({option.gpu_type}): {option.gpus} GPU(s), "
                    f"livres agora {option.free_now}, fila {option.pending}"
                )
    else:
        print("Nenhuma partição serve com essa configuração.")
        wider, _ = suggest(info, partitions, gpu_util=SUGGEST_WIDER_UTIL, max_len=args.max_len)
        if wider:
            names = ", ".join(f"{s.partition} ({s.gpus} GPU)" for s in wider)
            print(
                f"Com gpu-util {SUGGEST_WIDER_UTIL} caberia em: {names}. "
                "Isso exige a GPU quase toda livre, então pode falhar em GPU compartilhada."
            )

    if rejected:
        print("\nDescartadas:")
        for item in rejected:
            print(f"  {item.partition}: {item.reason}")
    return 0


def _start_command(model: str, partition: str, gpus: int, gpu_util: float, max_len: int) -> str:
    return (
        f"kitsune start --model {model} --partition {partition} --gpus {gpus} "
        f"--gpu-util {gpu_util} --max-len {max_len}"
    )


def cmd_chat(args: argparse.Namespace) -> int:
    text = " ".join(args.message).strip()
    if not text:
        raise ValueError("escreva uma mensagem")
    starts_new = args.new or args.system is not None
    if args.chat and starts_new:
        raise ValueError("--chat não combina com --new ou --system")

    store = ChatStore()
    if args.chat:
        conversation = store.load(args.chat)
    elif starts_new:
        conversation = store.new(system=args.system)
    else:
        conversation = store.active() or store.new()

    running = [job for job in list_jobs() if job.state == "RUNNING"]
    if not running:
        raise RemoteError("nenhum servidor no ar. Use `kitsune start` e acompanhe com `kitsune status`.")
    state = read_state(running[0].job_id)
    if state.get("STATUS") != "ready":
        label = STATUS_LABEL.get(state.get("STATUS", ""), "iniciando")
        raise RemoteError(f"o servidor ainda não está pronto ({label}). Veja `kitsune status`.")

    with tunnel(state["NODE"], int(state["PORT"])) as local_port:
        reply = chat_completion(
            local_port,
            state["TOKEN"],
            state["MODEL"],
            conversation.request_messages(text),
        )
    conversation.record(text, reply)
    store.save(conversation)
    store.set_active(conversation.id)
    print(reply)
    print(f"\n[chat {conversation.id}]", file=sys.stderr)
    return 0


def _raise_interrupt(*_: object) -> None:
    raise KeyboardInterrupt()


def cmd_tunnel(args: argparse.Namespace) -> int:
    sys.stdout.reconfigure(line_buffering=True)
    # Processos de fundo de shells não interativos herdam SIGINT ignorado; SIGTERM também fecha o túnel.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, _raise_interrupt)
    running = [job for job in list_jobs() if job.state == "RUNNING"]
    if not running:
        raise RemoteError("nenhum servidor no ar. Use `kitsune start` e acompanhe com `kitsune status`.")
    state = read_state(running[0].job_id)
    if state.get("STATUS") != "ready":
        label = STATUS_LABEL.get(state.get("STATUS", ""), "iniciando")
        raise RemoteError(f"o servidor ainda não está pronto ({label}). Veja `kitsune status`.")

    save_api_key(state["TOKEN"], config.API_KEY_FILE)
    with tunnel(state["NODE"], int(state["PORT"]), local_port=args.port) as local_port:
        print(f"Túnel aberto: http://127.0.0.1:{local_port}/v1 -> {state['NODE']}:{state['PORT']} (job {running[0].job_id})")
        print(f"Modelo: {state['MODEL']}")
        print(f"Chave: {config.API_KEY_FILE} (permissão 600)")
        print("\nEm outro terminal:")
        print(f'  export KITSUNE_API_KEY="$(cat {config.API_KEY_FILE})"')
        print(f"  interact --provider kitsune --model {state['MODEL']} --msg \"...\"")
        print("\nCtrl-C fecha o túnel.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nTúnel fechado.")
    return 0


def cmd_chats(args: argparse.Namespace) -> int:
    store = ChatStore()
    chats = store.list()
    if not chats:
        print('Nenhum chat ainda. Use `kitsune chat "..."`.')
        return 0
    active = store.active_id()
    print("  ID    CRIADO               TURNOS  PRIMEIRA MENSAGEM")
    for chat in chats:
        marker = "*" if chat.id == active else " "
        first = next((m["content"] for m in chat.messages if m["role"] == "user"), "")
        preview = " ".join(first.split())[:PREVIEW_CHARS] or "(vazio)"
        print(f"{marker} {chat.id}  {chat.created}  {chat.turns:>6}  {preview}")
    print("\n* = chat ativo (o que `kitsune chat` continua)")
    return 0


def cmd_history(args: argparse.Namespace) -> int:
    store = ChatStore()
    if args.chat_id:
        conversation = store.load(args.chat_id)
    else:
        conversation = store.active()
        if conversation is None:
            print('Nenhum chat ativo. Use `kitsune chat "..."` ou `kitsune chats`.')
            return 0
    print(f"Chat {conversation.id} (criado em {conversation.created})\n")
    if conversation.system:
        print(f"[sistema] {conversation.system}\n")
    if not conversation.messages:
        print("(sem mensagens)")
        return 0
    for message in conversation.messages:
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
        description="Mini-OpenRouter: sobe um modelo num nó de GPU do cluster por tempo limitado e conversa com ele.",
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

    suggest_cmd = sub.add_parser("suggest", help="sugere partição e número de GPUs para um modelo")
    suggest_cmd.add_argument("--model", default=config.DEFAULT_MODEL, help="autor/repo do Hugging Face")
    suggest_cmd.add_argument("--gpu-util", type=float, default=config.GPU_MEMORY_UTILIZATION)
    suggest_cmd.add_argument("--max-len", type=int, default=config.MAX_MODEL_LEN)
    suggest_cmd.set_defaults(func=cmd_suggest)

    start = sub.add_parser("start", help="sobe o modelo num nó de GPU")
    start.add_argument("--model", default=config.DEFAULT_MODEL, help="autor/repo do Hugging Face")
    start.add_argument("--partition", default=config.DEFAULT_PARTITION)
    start.add_argument("--gpus", type=int, default=config.DEFAULT_GPUS)
    start.add_argument("--cpus", type=int, default=config.DEFAULT_CPUS)
    start.add_argument("--mem", default=config.DEFAULT_MEM)
    start.add_argument("--time", default=config.DEFAULT_TIME, help="limite do job, HH:MM:SS")
    start.add_argument("--gpu-util", type=float, default=config.GPU_MEMORY_UTILIZATION)
    start.add_argument("--max-len", type=int, default=config.MAX_MODEL_LEN)
    start.set_defaults(func=cmd_start)

    status = sub.add_parser("status", help="mostra se há um servidor no ar")
    status.set_defaults(func=cmd_status)

    chat = sub.add_parser("chat", help="manda uma mensagem ao modelo")
    chat.add_argument("message", nargs="+")
    chat.add_argument("-c", "--chat", metavar="ID", help="continua o chat ID (e o torna o ativo)")
    chat.add_argument("--new", action="store_true", help="começa um chat novo")
    chat.add_argument("--system", default=None, help="instrução de sistema; começa um chat novo")
    chat.set_defaults(func=cmd_chat)

    tunnel_cmd = sub.add_parser("tunnel", help="abre o túnel até o servidor, para outras ferramentas (ex.: interact)")
    tunnel_cmd.add_argument("--port", type=int, default=config.TUNNEL_PORT, help="porta local (padrão: %(default)s)")
    tunnel_cmd.set_defaults(func=cmd_tunnel)

    chats = sub.add_parser("chats", help="lista os chats guardados")
    chats.set_defaults(func=cmd_chats)

    history = sub.add_parser("history", help="mostra um chat (o ativo, se não passar ID)")
    history.add_argument("chat_id", nargs="?", metavar="ID")
    history.set_defaults(func=cmd_history)

    stop = sub.add_parser("stop", help="cancela o servidor no ar")
    stop.set_defaults(func=cmd_stop)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (RemoteError, ChatError, ValueError) as error:
        print(f"erro: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130

"""Túnel ssh até o vLLM do nó e chamada à API no formato OpenAI (o mesmo do OpenRouter)."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .config import HOST


class ChatError(RuntimeError):
    pass


def _free_local_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextmanager
def tunnel(
    node: str,
    port: int,
    *,
    host: str = HOST,
    timeout: float = 20.0,
    local_port: int | None = None,
) -> Iterator[int]:
    if local_port is None:
        local_port = _free_local_port()
    forward = f"127.0.0.1:{local_port}:{node}:{port}"
    process = subprocess.Popen(["ssh", "-N", "-o", "ExitOnForwardFailure=yes", "-L", forward, host])
    try:
        deadline = time.monotonic() + timeout
        while True:
            if process.poll() is not None:
                raise ChatError("o túnel ssh encerrou antes de abrir")
            try:
                with socket.create_connection(("127.0.0.1", local_port), timeout=0.5):
                    break
            except OSError:
                if time.monotonic() > deadline:
                    raise ChatError("o túnel ssh não abriu a tempo")
                time.sleep(0.2)
        yield local_port
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        # Com ControlMaster, o forward fica no master depois que o cliente sai; cancela explicitamente.
        subprocess.run(["ssh", "-O", "cancel", "-L", forward, host], capture_output=True)


def save_api_key(token: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(token + "\n")


def chat_completion(
    local_port: int,
    token: str,
    model: str,
    messages: list[dict[str, str]],
    *,
    timeout: float = 300.0,
) -> str:
    body = json.dumps({"model": model, "messages": messages}).encode("utf-8")
    request = urllib.request.Request(
        f"http://127.0.0.1:{local_port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise ChatError(f"o servidor respondeu {error.code}: {detail}") from error
    except OSError as error:
        raise ChatError(f"não consegui falar com o servidor: {error}") from error
    try:
        return payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise ChatError(f"resposta inesperada do servidor: {payload}") from error

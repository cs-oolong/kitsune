"""Túnel ssh até o vLLM do nó."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .config import HOST
from .remote import RemoteError


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
                raise RemoteError("o túnel ssh encerrou antes de abrir")
            try:
                with socket.create_connection(("127.0.0.1", local_port), timeout=0.5):
                    break
            except OSError:
                if time.monotonic() > deadline:
                    raise RemoteError("o túnel ssh não abriu a tempo")
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


def save_tunnel_info(info: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(json.dumps(info) + "\n")


def read_tunnel_info(path: Path) -> dict | None:
    try:
        info = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return info if isinstance(info, dict) else None


def clear_tunnel_info(path: Path) -> None:
    path.unlink(missing_ok=True)


def probe_shared_tunnel(info: dict, token: str, timeout: float = 1.5) -> str | None:
    """Base URL do túnel compartilhado, se ele estiver vivo e for do MESMO servidor.

    A validação é uma chamada autenticada a /v1/models: recusa (401) significa
    túnel de um servidor anterior; conexão recusada, túnel morto.
    """
    try:
        local_port = int(info["local_port"])
    except (KeyError, TypeError, ValueError):
        return None
    base = f"http://127.0.0.1:{local_port}/v1"
    request = urllib.request.Request(
        f"{base}/models", headers={"Authorization": f"Bearer {token}"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status == 200:
                return base
    except OSError:
        pass
    return None

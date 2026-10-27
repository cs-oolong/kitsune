import fcntl
import json
import os
import re
import secrets
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from .errors import ChatError

SCHEMA_VERSION = 1
CHAT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def ensure_ledger_dir(ledger_dir: Path) -> None:
    (ledger_dir / ".locks").mkdir(parents=True, exist_ok=True)


def validate_chat_id(chat_id: str) -> None:
    if not CHAT_ID_RE.match(chat_id):
        raise ChatError(
            "chat_id must start with a letter/digit and contain only "
            "letters, digits, '.', '_' or '-' (max 128 chars)",
            code="invalid_chat_id",
        )


def new_chat_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return "chat-{}-{}".format(stamp, secrets.token_hex(3))


def chat_path(ledger_dir: Path, chat_id: str) -> Path:
    return ledger_dir / (chat_id + ".json")


def load_chat(ledger_dir: Path, chat_id: str) -> Optional[Dict[str, Any]]:
    path = chat_path(ledger_dir, chat_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ChatError(
            "could not read chat {}: {}".format(path, exc),
            code="ledger_corrupt",
        )


def save_chat(ledger_dir: Path, chat: Dict[str, Any]) -> None:
    path = chat_path(ledger_dir, chat["chat_id"])
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(chat, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


@contextmanager
def chat_lock(ledger_dir: Path, chat_id: str) -> Iterator[None]:
    """Serialize turns for one chat. Different chats use different locks."""
    lock_path = ledger_dir / ".locks" / (chat_id + ".lock")
    with open(lock_path, "a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield


def history(
    chat: Dict[str, Any], include_errors: bool = True
) -> List[Dict[str, Any]]:
    """History for the model (errors skipped) or the CLI response (errors shown)."""
    out: List[Dict[str, Any]] = []
    for msg in chat.get("messages", []):
        if msg.get("role") == "user":
            out.append({"role": "user", "content": msg.get("content") or ""})
        elif msg.get("error") is not None:
            if include_errors:
                out.append({"role": "assistant", "error": msg["error"]})
        else:
            out.append({"role": "assistant", "content": msg.get("content") or ""})
    return out

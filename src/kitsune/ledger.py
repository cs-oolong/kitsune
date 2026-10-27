"""Leitura do ledger do chat-with-ledger (ledger/*.json, o mesmo formato que o interact grava)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_chats(ledger_dir: Path) -> list[dict[str, Any]]:
    if not ledger_dir.is_dir():
        return []
    chats = []
    for path in ledger_dir.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and "chat_id" in data:
            chats.append(data)
    return sorted(chats, key=lambda chat: chat.get("updated_at", ""), reverse=True)


def latest_chat(ledger_dir: Path, *, provider: str, model_id: str | None = None) -> str | None:
    for chat in read_chats(ledger_dir):
        if chat.get("provider") != provider:
            continue
        if model_id is not None and chat.get("model_id") != model_id:
            continue
        return chat["chat_id"]
    return None

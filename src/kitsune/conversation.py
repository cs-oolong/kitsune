"""Chats guardados só no laptop: um arquivo JSON por chat, com ID curto. O servidor não guarda nada."""

from __future__ import annotations

import json
import re
import secrets
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .config import CHATS_DIR

CHAT_ID_PATTERN = re.compile(r"[0-9a-f]{4}")
ACTIVE_FILE = "active"


@dataclass
class Conversation:
    id: str
    created: str
    system: str | None = None
    messages: list[dict[str, str]] = field(default_factory=list)

    def request_messages(self, user_text: str) -> list[dict[str, str]]:
        messages = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.extend(self.messages)
        messages.append({"role": "user", "content": user_text})
        return messages

    def record(self, user_text: str, reply: str) -> None:
        self.messages.append({"role": "user", "content": user_text})
        self.messages.append({"role": "assistant", "content": reply})

    @property
    def turns(self) -> int:
        return sum(1 for message in self.messages if message["role"] == "user")


class ChatStore:
    def __init__(self, directory: Path = CHATS_DIR) -> None:
        self.directory = directory

    def new(self, system: str | None = None) -> Conversation:
        chat_id = self._new_id()
        created = datetime.now().isoformat(timespec="seconds")
        return Conversation(id=chat_id, created=created, system=system)

    def load(self, chat_id: str) -> Conversation:
        path = self._path(chat_id)
        if not path.exists():
            raise ValueError(f"chat {chat_id} não existe. Veja `kitsune chats`.")
        data = json.loads(path.read_text(encoding="utf-8"))
        return Conversation(
            id=data["id"],
            created=data["created"],
            system=data.get("system"),
            messages=list(data.get("messages", [])),
        )

    def save(self, conversation: Conversation) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "id": conversation.id,
            "created": conversation.created,
            "system": conversation.system,
            "messages": conversation.messages,
        }
        path = self._path(conversation.id)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def list(self) -> list[Conversation]:
        if not self.directory.exists():
            return []
        chats = [self.load(path.stem) for path in self.directory.glob("*.json")]
        return sorted(chats, key=lambda chat: chat.created, reverse=True)

    def active_id(self) -> str | None:
        path = self.directory / ACTIVE_FILE
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8").strip() or None

    def active(self) -> Conversation | None:
        chat_id = self.active_id()
        if chat_id is None or not self._path(chat_id).exists():
            return None
        return self.load(chat_id)

    def set_active(self, chat_id: str) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / ACTIVE_FILE).write_text(chat_id + "\n", encoding="utf-8")

    def _path(self, chat_id: str) -> Path:
        if not CHAT_ID_PATTERN.fullmatch(chat_id):
            raise ValueError(f"ID de chat inválido: {chat_id!r} (são 4 caracteres hexadecimais)")
        return self.directory / f"{chat_id}.json"

    def _new_id(self) -> str:
        while True:
            chat_id = secrets.token_hex(2)
            if not self._path(chat_id).exists():
                return chat_id

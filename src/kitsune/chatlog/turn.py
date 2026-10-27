import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .. import __version__
from .errors import ChatError
from .ledger import (
    SCHEMA_VERSION,
    chat_lock,
    ensure_ledger_dir,
    history,
    load_chat,
    new_chat_id,
    save_chat,
    utc_now,
    validate_chat_id,
)
from .providers import DEFAULT_PROVIDER, call_model, get_provider

DEFAULT_MAX_TOKENS = 2048
DEFAULT_TIMEOUT_S = 180
RETRY_DELAYS_S = (5, 15, 30)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_system_prompt(path: Path) -> str:
    """Read the whole file verbatim. Only used when a chat is created."""
    if not path.is_file():
        raise ChatError(
            "system prompt file not found: {}".format(path),
            code="missing_system_prompt",
        )
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ChatError(
            "system prompt file is empty: {}".format(path),
            code="missing_system_prompt",
        )
    return text


def run_turn(
    model_id: str,
    message: str,
    *,
    chat_id: Optional[Any] = None,
    ledger_dir: Path = Path("ledger"),
    system_prompt_file: Path = Path("system_prompt.txt"),
    provider_name: str = DEFAULT_PROVIDER,
    base_url: Optional[str] = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: int = DEFAULT_TIMEOUT_S,
    retry_delays: tuple = RETRY_DELAYS_S,
) -> Dict[str, Any]:
    """Append one user turn, ask the model, append the reply, save.

    A new chat snapshots the system prompt file (and its sha256) into the
    ledger. Existing chats keep their snapshot, whatever the file says now.
    A chat is bound to one provider and one model; switching is rejected.
    """
    if not isinstance(model_id, str) or not model_id.strip():
        raise ChatError(
            "model_id must be a non-empty string", code="invalid_model_id"
        )
    if not isinstance(message, str) or not message.strip():
        raise ChatError(
            "message must be a non-empty string", code="invalid_message"
        )
    model_id = model_id.strip()
    provider = get_provider(provider_name)
    base = (base_url or provider.base_url).rstrip("/")

    ensure_ledger_dir(ledger_dir)
    if chat_id is None or chat_id == "":
        chat_id = new_chat_id()
    else:
        chat_id = str(chat_id)
        validate_chat_id(chat_id)

    with chat_lock(ledger_dir, chat_id):
        chat = load_chat(ledger_dir, chat_id)
        if chat is None:
            system_prompt = read_system_prompt(Path(system_prompt_file))
            now = utc_now()
            chat = {
                "schema_version": SCHEMA_VERSION,
                "chat_id": chat_id,
                "provider": provider.name,
                "model_id": model_id,
                "system_prompt": system_prompt,
                "system_prompt_sha256": sha256_text(system_prompt),
                "created_at": now,
                "updated_at": now,
                "messages": [],
            }
        else:
            existing = (chat.get("provider", DEFAULT_PROVIDER), chat.get("model_id"))
            if existing != (provider.name, model_id):
                raise ChatError(
                    "chat {} belongs to {}/{}, not {}/{}".format(
                        chat_id, existing[0], existing[1], provider.name, model_id
                    ),
                    code="model_mismatch",
                )
        system_prompt = chat["system_prompt"]

        chat["messages"].append(
            {"role": "user", "content": message, "created_at": utc_now()}
        )

        prompt: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]
        prompt.extend(history(chat, include_errors=False))
        sent_sha256 = sha256_text(
            json.dumps(prompt, ensure_ascii=False, sort_keys=True)
        )

        result = call_model(
            provider,
            model_id,
            prompt,
            base_url=base,
            max_tokens=max_tokens,
            timeout=timeout,
            retry_delays=retry_delays,
        )

        if result.get("error") is not None:
            chat["messages"].append(
                {
                    "role": "assistant",
                    "error": result["error"],
                    "created_at": utc_now(),
                }
            )
        else:
            meta = dict(result.get("meta") or {})
            meta.update(
                {
                    "base_url": base,
                    "sent_sha256": sent_sha256,
                    "tool": {"name": "kitsune", "version": __version__},
                }
            )
            chat["messages"].append(
                {
                    "role": "assistant",
                    "content": result.get("content") or "",
                    "created_at": utc_now(),
                    "meta": meta,
                }
            )

        chat["updated_at"] = utc_now()
        save_chat(ledger_dir, chat)

    return {
        "chat_id": chat_id,
        "provider": provider.name,
        "model_id": model_id,
        "system_prompt": system_prompt,
        "history": history(chat),
    }

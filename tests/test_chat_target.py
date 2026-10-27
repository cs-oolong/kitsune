import json
from pathlib import Path

import pytest

from kitsune.chatlog.errors import ChatError
from kitsune.cli import check_chat_target


def write_chat(ledger_dir: Path, chat_id: str) -> None:
    ledger_dir.mkdir(parents=True, exist_ok=True)
    (ledger_dir / f"{chat_id}.json").write_text(json.dumps({"chat_id": chat_id, "messages": []}), encoding="utf-8")


def test_no_id_means_new_chat(tmp_path):
    check_chat_target(tmp_path / "ledger-kitsune", None, None)
    check_chat_target(tmp_path / "ledger-kitsune", None, "instrução")


def test_existing_id_continues(tmp_path):
    write_chat(tmp_path, "chat-1")
    check_chat_target(tmp_path, "chat-1", None)


def test_unknown_id_is_an_error(tmp_path):
    with pytest.raises(ChatError) as exc:
        check_chat_target(tmp_path, "nao-existe", None)
    assert exc.value.code == "chat_not_found"


def test_system_with_id_is_an_error(tmp_path):
    write_chat(tmp_path, "chat-1")
    with pytest.raises(ChatError) as exc:
        check_chat_target(tmp_path, "chat-1", "outro prompt")
    assert exc.value.code == "bad_request"


def test_invalid_id_is_an_error(tmp_path):
    with pytest.raises(ChatError) as exc:
        check_chat_target(tmp_path, "../fora", None)
    assert exc.value.code == "invalid_chat_id"

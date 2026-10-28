import json

from kitsune.chatlog.ledger import latest_chat, read_chats


def write_chat(ledger_dir, chat_id, provider, model_id, updated_at):
    ledger_dir.mkdir(parents=True, exist_ok=True)
    data = {"chat_id": chat_id, "provider": provider, "model_id": model_id, "updated_at": updated_at, "messages": []}
    (ledger_dir / f"{chat_id}.json").write_text(json.dumps(data), encoding="utf-8")


def test_read_chats_is_newest_first_and_ignores_other_files(tmp_path):
    ledger = tmp_path / "ledger"
    write_chat(ledger, "old", "kitsune", "Qwen", "2026-10-27T10:00:00Z")
    write_chat(ledger, "new", "kitsune", "Qwen", "2026-10-27T11:00:00Z")
    (ledger / "notes.txt").write_text("ignorado", encoding="utf-8")
    (ledger / "broken.json").write_text("{não é json", encoding="utf-8")
    assert [c["chat_id"] for c in read_chats(ledger)] == ["new", "old"]


def test_read_chats_missing_dir_is_empty(tmp_path):
    assert read_chats(tmp_path / "nope") == []


def test_latest_chat_only_picks_the_same_provider_and_model(tmp_path):
    ledger = tmp_path / "ledger"
    write_chat(ledger, "other-provider-recent", "outro-provedor", "Qwen", "2026-10-27T12:00:00Z")
    write_chat(ledger, "other-model", "kitsune", "Ministral", "2026-10-27T11:30:00Z")
    write_chat(ledger, "kitsune-old", "kitsune", "Qwen", "2026-10-27T10:00:00Z")
    assert latest_chat(ledger, provider="kitsune", model_id="Qwen") == "kitsune-old"


def test_latest_chat_none_when_nothing_matches(tmp_path):
    assert latest_chat(tmp_path / "ledger", provider="kitsune") is None

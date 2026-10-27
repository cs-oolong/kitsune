import pytest

from kitsune.conversation import ChatStore


@pytest.fixture
def store(tmp_path):
    return ChatStore(tmp_path / "chats")


def test_new_chat_has_short_hex_id_and_is_not_saved_until_recorded(store):
    conversation = store.new(system="Responda em português.")
    assert len(conversation.id) == 4
    assert all(c in "0123456789abcdef" for c in conversation.id)
    assert store.list() == []


def test_request_messages_puts_system_first_and_appends_user_turn(store):
    conversation = store.new(system="Responda em português.")
    conversation.record("oi", "olá!")
    assert conversation.request_messages("tudo bem?") == [
        {"role": "system", "content": "Responda em português."},
        {"role": "user", "content": "oi"},
        {"role": "assistant", "content": "olá!"},
        {"role": "user", "content": "tudo bem?"},
    ]


def test_request_messages_does_not_change_history(store):
    conversation = store.new()
    conversation.request_messages("pergunta")
    assert conversation.messages == []


def test_chats_are_kept_side_by_side(store):
    first = store.new()
    first.record("primeiro", "r1")
    store.save(first)
    second = store.new()
    second.record("segundo", "r2")
    store.save(second)

    assert store.load(first.id).messages == first.messages
    assert store.load(second.id).messages == second.messages
    assert {chat.id for chat in store.list()} == {first.id, second.id}


def test_active_pointer_follows_set_active(store):
    conversation = store.new()
    store.save(conversation)
    store.set_active(conversation.id)
    assert store.active_id() == conversation.id
    assert store.active().id == conversation.id


def test_active_is_none_when_nothing_was_saved(store):
    assert store.active_id() is None
    assert store.active() is None


def test_active_is_none_when_pointed_chat_is_gone(store):
    conversation = store.new()
    store.save(conversation)
    store.set_active(conversation.id)
    store._path(conversation.id).unlink()
    assert store.active() is None


def test_load_missing_chat_is_an_error(store):
    with pytest.raises(ValueError, match="não existe"):
        store.load("beef")


@pytest.mark.parametrize("bad", ["../x", "ABCD", "abc", "abcde", ""])
def test_chat_ids_are_validated(store, bad):
    with pytest.raises(ValueError, match="inválido"):
        store.load(bad)


def test_list_is_newest_first(store):
    older = store.new()
    older.created = "2026-10-27T10:00:00"
    store.save(older)
    newer = store.new()
    newer.created = "2026-10-27T11:00:00"
    store.save(newer)
    assert [chat.id for chat in store.list()] == [newer.id, older.id]


def test_turns_count_user_messages(store):
    conversation = store.new()
    conversation.record("a", "b")
    conversation.record("c", "d")
    assert conversation.turns == 2

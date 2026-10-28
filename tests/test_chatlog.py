import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from kitsune.chatlog.ledger import load_chat
from kitsune.chatlog.turn import run_turn


class FakeServer(BaseHTTPRequestHandler):
    replies: list = []
    requests: list = []

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        FakeServer.requests.append(json.loads(self.rfile.read(length)))
        body = json.dumps(
            {"model": "Qwen2.5-0.5B-Instruct", "choices": [{"message": {"content": FakeServer.replies.pop(0)}, "finish_reason": "stop"}]}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setenv("KITSUNE_API_KEY", "token")
    FakeServer.replies = ["primeira", "segunda"]
    FakeServer.requests = []
    httpd = HTTPServer(("127.0.0.1", 0), FakeServer)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}/v1"
    httpd.shutdown()


def test_new_chat_is_recorded_in_the_ledger_format(tmp_path, server):
    prompt = tmp_path / "system_prompt.txt"
    prompt.write_text("Responda em português.", encoding="utf-8")
    ledger = tmp_path / "ledger-kitsune"

    result = run_turn("Qwen2.5-0.5B-Instruct", "oi", ledger_dir=ledger, system_prompt_file=prompt,
                      provider_name="kitsune", base_url=server, retry_delays=(0, 0))

    chat = load_chat(ledger, result["chat_id"])
    assert chat["provider"] == "kitsune"
    assert chat["model_id"] == "Qwen2.5-0.5B-Instruct"
    assert chat["messages"][1]["content"] == "primeira"
    assert chat["messages"][1]["meta"]["tool"]["name"] == "kitsune"
    assert chat["messages"][1]["meta"]["base_url"] == server


def test_max_tokens_is_recorded_in_meta(tmp_path, server):
    prompt = tmp_path / "system_prompt.txt"
    prompt.write_text("Responda em português.", encoding="utf-8")
    ledger = tmp_path / "ledger-kitsune"

    result = run_turn("Qwen2.5-0.5B-Instruct", "oi", ledger_dir=ledger, system_prompt_file=prompt,
                      provider_name="kitsune", base_url=server, max_tokens=1234, retry_delays=(0, 0))

    chat = load_chat(ledger, result["chat_id"])
    assert chat["messages"][1]["meta"]["max_tokens"] == 1234


def test_continuation_sends_the_history(tmp_path, server):
    prompt = tmp_path / "system_prompt.txt"
    prompt.write_text("Responda em português.", encoding="utf-8")
    ledger = tmp_path / "ledger-kitsune"
    first = run_turn("Qwen2.5-0.5B-Instruct", "um", ledger_dir=ledger, system_prompt_file=prompt,
                     provider_name="kitsune", base_url=server, retry_delays=(0, 0))
    run_turn("Qwen2.5-0.5B-Instruct", "dois", chat_id=first["chat_id"], ledger_dir=ledger,
             system_prompt_file=prompt, provider_name="kitsune", base_url=server, retry_delays=(0, 0))

    sent = FakeServer.requests[1]["messages"]
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "user"]
    assert sent[2]["content"] == "primeira"

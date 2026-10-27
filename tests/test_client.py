import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import os
import stat

import pytest

from kitsune.client import ChatError, chat_completion, save_api_key


def test_save_api_key_is_private_to_the_user(tmp_path):
    path = tmp_path / "state" / "api_key"
    save_api_key("segredo", path)
    assert path.read_text() == "segredo\n"
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600


def test_save_api_key_replaces_an_old_token(tmp_path):
    path = tmp_path / "api_key"
    save_api_key("antigo", path)
    save_api_key("novo", path)
    assert path.read_text() == "novo\n"


class FakeVLLM(BaseHTTPRequestHandler):
    seen: dict = {}

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        FakeVLLM.seen = {
            "path": self.path,
            "auth": self.headers.get("Authorization"),
            "body": json.loads(self.rfile.read(length)),
        }
        if self.headers.get("Authorization") != "Bearer segredo":
            self.send_response(401)
            self.end_headers()
            self.wfile.write(b'{"error": "unauthorized"}')
            return
        reply = {"choices": [{"message": {"role": "assistant", "content": "oi, humano"}}]}
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(reply).encode())

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    httpd = HTTPServer(("127.0.0.1", 0), FakeVLLM)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd.server_address[1]
    httpd.shutdown()


def test_chat_completion_sends_bearer_token_and_returns_content(server):
    messages = [{"role": "user", "content": "oi"}]
    reply = chat_completion(server, "segredo", "Qwen2.5-0.5B-Instruct", messages)
    assert reply == "oi, humano"
    assert FakeVLLM.seen["path"] == "/v1/chat/completions"
    assert FakeVLLM.seen["auth"] == "Bearer segredo"
    assert FakeVLLM.seen["body"] == {"model": "Qwen2.5-0.5B-Instruct", "messages": messages}


def test_chat_completion_reports_rejected_token(server):
    with pytest.raises(ChatError, match="401"):
        chat_completion(server, "errado", "Qwen2.5-0.5B-Instruct", [])

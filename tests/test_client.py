import json
import os
import stat
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

from kitsune.client import (
    clear_tunnel_info,
    probe_shared_tunnel,
    read_tunnel_info,
    save_api_key,
    save_tunnel_info,
)


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


def test_tunnel_info_roundtrip_is_private_to_the_user(tmp_path):
    path = tmp_path / "state" / "tunnel.json"
    info = {"job_id": "121903", "node": "dl-06", "port": "36865", "local_port": 8000}
    save_tunnel_info(info, path)
    assert read_tunnel_info(path) == info
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    clear_tunnel_info(path)
    assert read_tunnel_info(path) is None


def test_read_tunnel_info_tolerates_missing_or_garbage(tmp_path):
    assert read_tunnel_info(tmp_path / "nada.json") is None
    garbage = tmp_path / "tunnel.json"
    garbage.write_text("não é json", encoding="utf-8")
    assert read_tunnel_info(garbage) is None
    garbage.write_text("[1, 2]", encoding="utf-8")
    assert read_tunnel_info(garbage) is None


class _FakeVllm(BaseHTTPRequestHandler):
    TOKEN = "segredo"

    def do_GET(self):
        ok = self.headers.get("Authorization") == f"Bearer {self.TOKEN}"
        self.send_response(200 if ok else 401)
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, *args):
        pass


def _serve():
    server = HTTPServer(("127.0.0.1", 0), _FakeVllm)
    Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_probe_shared_tunnel_accepts_the_same_server():
    server = _serve()
    try:
        port = server.server_address[1]
        base = probe_shared_tunnel({"local_port": port}, "segredo")
        assert base == f"http://127.0.0.1:{port}/v1"
    finally:
        server.shutdown()


def test_probe_shared_tunnel_rejects_another_servers_tunnel():
    server = _serve()
    try:
        port = server.server_address[1]
        assert probe_shared_tunnel({"local_port": port}, "token-de-outro-job") is None
    finally:
        server.shutdown()


def test_probe_shared_tunnel_rejects_dead_tunnel_and_bad_info():
    assert probe_shared_tunnel({"local_port": 1}, "segredo", timeout=0.3) is None
    assert probe_shared_tunnel({}, "segredo", timeout=0.3) is None
    assert probe_shared_tunnel({"local_port": "não-número"}, "segredo", timeout=0.3) is None

import os
import stat

from kitsune.client import save_api_key


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

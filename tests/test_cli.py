import kitsune.cli as cli
from kitsune.cli import collect_status, extract_cause
from kitsune.remote import Job, LastJob

HARMONY_LOG = """\x1b[1;36m(APIServer pid=616260)\x1b[0m Traceback (most recent call last):
\x1b[1;36m(APIServer pid=616260)\x1b[0m   File "/home/ana/kitsune/venv/bin/vllm", line 10, in <module>
\x1b[1;36m(APIServer pid=616260)\x1b[0m     sys.exit(main())
\x1b[1;36m(APIServer pid=616260)\x1b[0m               ^^^^^^
\x1b[1;36m(APIServer pid=616260)\x1b[0m   File "/home/ana/kitsune/venv/lib/python3.12/site-packages/openai_harmony/__init__.py", line 699, in load_harmony_encoding
\x1b[1;36m(APIServer pid=616260)\x1b[0m     inner: _PyHarmonyEncoding = _load_harmony_encoding(name)
\x1b[1;36m(APIServer pid=616260)\x1b[0m                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
\x1b[1;36m(APIServer pid=616260)\x1b[0m openai_harmony.HarmonyError: error downloading or loading vocab file: failed to download or load vocab file
"""


def test_extract_cause_returns_the_exception_line_of_the_last_traceback():
    cause = extract_cause(HARMONY_LOG)
    assert cause == "openai_harmony.HarmonyError: error downloading or loading vocab file: failed to download or load vocab file"


def test_extract_cause_uses_the_last_traceback_when_chained():
    log = HARMONY_LOG + "Traceback (most recent call last):\n  File \"x.py\", line 1, in <module>\n    boom()\nRuntimeError: segunda falha\n"
    assert extract_cause(log) == "RuntimeError: segunda falha"


def test_extract_cause_falls_back_to_the_last_errorish_line():
    log = "INFO tudo bem\nERROR 10-28 12:14 out of memory no device 0\nINFO encerrando\n"
    assert extract_cause(log) == "ERROR 10-28 12:14 out of memory no device 0"


def test_extract_cause_none_when_nothing_looks_like_an_error():
    assert extract_cause("INFO começou\nINFO terminou\n") is None
    assert extract_cause("") is None


def fake_live_job(monkeypatch, state_file):
    monkeypatch.setattr(cli, "list_jobs", lambda: [Job("121900", "RUNNING", "dl-06", "0:10")])
    monkeypatch.setattr(cli, "read_state", lambda job_id: state_file)


def test_collect_status_ready_when_the_state_says_ready(monkeypatch):
    fake_live_job(monkeypatch, {"STATUS": "ready", "NODE": "dl-06"})
    view = collect_status()
    assert view.situation == "ready"
    assert "pronto" in view.lines[0]


def test_collect_status_waits_while_loading_or_pending(monkeypatch):
    fake_live_job(monkeypatch, {"STATUS": "loading", "NODE": "dl-06"})
    assert collect_status().situation == "waiting"
    monkeypatch.setattr(cli, "list_jobs", lambda: [Job("121900", "PENDING", "Resources", "0:00")])
    view = collect_status()
    assert view.situation == "waiting"
    assert "pending" in view.lines[0]


def test_collect_status_failed_when_the_state_says_failed(monkeypatch):
    fake_live_job(monkeypatch, {"STATUS": "failed", "NODE": "dl-06"})
    view = collect_status()
    assert view.situation == "failed"
    assert "kitsune logs 121900" in view.lines[0]


def test_collect_status_empty_when_no_job_ever(monkeypatch):
    monkeypatch.setattr(cli, "list_jobs", lambda: [])
    monkeypatch.setattr(cli, "last_job", lambda: None)
    view = collect_status()
    assert view.situation == "empty"
    assert view.lines == ["Nenhum servidor no ar. Use `kitsune start`."]


def test_collect_status_reports_cause_of_the_last_failed_job(monkeypatch):
    monkeypatch.setattr(cli, "list_jobs", lambda: [])
    monkeypatch.setattr(cli, "last_job", lambda: LastJob("121897", "FAILED", "1:0", {}))
    monkeypatch.setattr(cli, "read_log", lambda job_id, lines: (job_id, HARMONY_LOG))
    view = collect_status()
    assert view.situation == "failed"
    assert "falhou (saída 1:0)" in view.lines[0]
    assert any("HarmonyError" in line for line in view.lines)


def test_collect_status_reports_a_cancelled_last_job(monkeypatch):
    monkeypatch.setattr(cli, "list_jobs", lambda: [])
    monkeypatch.setattr(cli, "last_job", lambda: LastJob("121897", "CANCELLED by 12345", "0:0", {}))
    view = collect_status()
    assert view.situation == "ended"
    assert "foi cancelado" in view.lines[0]

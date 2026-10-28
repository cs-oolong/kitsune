from kitsune.cli import extract_cause

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

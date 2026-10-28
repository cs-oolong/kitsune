from kitsune.remote import parse_models, parse_squeue, parse_state, setup_script


def test_parse_models_reads_one_model_per_line_sorted_by_folder():
    text = (
        "Qwen2.5-7B-Instruct|Qwen/Qwen2.5-7B-Instruct|15G|sim\n"
        "Qwen2.5-0.5B-Instruct|?|1.0G|nao\n"
    )
    models = parse_models(text)
    assert [m.folder for m in models] == ["Qwen2.5-0.5B-Instruct", "Qwen2.5-7B-Instruct"]
    assert models[0].repo == "?"
    assert models[0].complete is False
    assert models[1].repo == "Qwen/Qwen2.5-7B-Instruct"
    assert models[1].size == "15G"
    assert models[1].complete is True


def test_parse_models_ignores_noise_and_empty_output():
    assert parse_models("") == []
    assert parse_models("linha sem separador\n") == []


def test_setup_script_records_the_repo_id_next_to_the_weights():
    script = setup_script("Qwen/Qwen2.5-0.5B-Instruct")
    assert 'echo "$MODEL" > "$MODEL_DIR/.kitsune-repo"' in script


def test_parse_squeue_reads_running_and_pending_jobs():
    text = "1234|RUNNING|dl-02|0:12\n1235|PENDING|Resources|0:00\n"
    jobs = parse_squeue(text)
    assert [(j.job_id, j.state, j.where, j.elapsed) for j in jobs] == [
        ("1234", "RUNNING", "dl-02", "0:12"),
        ("1235", "PENDING", "Resources", "0:00"),
    ]


def test_parse_squeue_empty_output_means_no_jobs():
    assert parse_squeue("") == []
    assert parse_squeue("\n") == []


def test_parse_state_reads_key_value_lines():
    text = "JOBID=1234\nNODE=dl-02\nPORT=41871\nTOKEN=abc=def\nSTATUS=ready\n"
    state = parse_state(text)
    assert state["NODE"] == "dl-02"
    assert state["PORT"] == "41871"
    assert state["TOKEN"] == "abc=def"
    assert state["STATUS"] == "ready"


def test_setup_script_downloads_into_the_models_folder():
    script = setup_script("Qwen/Qwen2.5-0.5B-Instruct")
    assert "snapshot_download(repo_id='$MODEL'" in script
    assert "$ROOT/models/Qwen2.5-0.5B-Instruct" in script
    assert "vllm==0.11.0" in script
    assert "transformers==4.55.2" in script
    assert "torch.__version__.startswith('2.8.0')" in script
    assert "float(torch.version.cuda) <= 12.8" in script
    assert 'export TIKTOKEN_RS_CACHE_DIR="$ROOT/tiktoken-cache"' in script
    assert "load_harmony_encoding('HarmonyGptOss')" in script
    assert "mv \"$ROOT/venv\" \"$ROOT/venv.bak.$(date +%s)\"" in script

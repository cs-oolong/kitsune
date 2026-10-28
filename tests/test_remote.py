import pytest

from kitsune.remote import parse_last_job, parse_models, parse_squeue, parse_state, read_log, setup_script


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


def test_parse_last_job_combines_sacct_and_state_of_the_same_job():
    text = "121897|FAILED|1:0\nJOBID=121897\nSTATUS=failed\nMODEL=gpt-oss-20b\n"
    report = parse_last_job(text)
    assert report is not None
    assert report.job_id == "121897"
    assert report.slurm_state == "FAILED"
    assert report.exit_code == "1:0"
    assert report.state["STATUS"] == "failed"


def test_parse_last_job_discards_state_from_an_older_job():
    text = "121900|FAILED|1:0\nJOBID=121897\nSTATUS=ready\n"
    report = parse_last_job(text)
    assert report is not None
    assert report.job_id == "121900"
    assert report.state == {}


def test_parse_last_job_without_accounting_falls_back_to_the_state_file():
    report = parse_last_job("JOBID=121897\nSTATUS=ready\nNODE=dl-06\n")
    assert report is not None
    assert report.job_id == "121897"
    assert report.slurm_state is None
    assert report.state["NODE"] == "dl-06"


def test_parse_last_job_keeps_the_full_cancelled_reason():
    report = parse_last_job("121897|CANCELLED by 12345|0:0\n")
    assert report is not None
    assert report.slurm_state == "CANCELLED by 12345"


def test_parse_last_job_empty_means_no_job_ever():
    assert parse_last_job("") is None
    assert parse_last_job("ruído sem separador\n") is None


def test_read_log_rejects_an_unsafe_job_id_before_touching_ssh():
    with pytest.raises(ValueError):
        read_log("1;rm -rf ~", 10)

import pytest

from kitsune.job import render_job_script
from kitsune.remote import validate_model


def render(
    model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    *,
    gpus: int = 1,
    dtype: str = "auto",
    trust_remote_code: bool = False,
) -> str:
    return render_job_script(
        model,
        partition="l40s",
        gpus=gpus,
        cpus=4,
        mem="16G",
        time="01:00:00",
        dtype=dtype,
        gpu_util=0.3,
        max_len=4096,
        trust_remote_code=trust_remote_code,
    )


def test_sbatch_directives_follow_arguments():
    script = render()
    assert "#SBATCH --partition=l40s" in script
    assert "#SBATCH --gres=gpu:1" in script
    assert "#SBATCH --mem=16G" in script
    assert "#SBATCH --time=01:00:00" in script
    assert "#SBATCH --output=logs/%j.out" in script


def test_serves_the_model_from_the_remote_models_folder():
    script = render()
    assert 'MODEL_DIR="$ROOT/models/Qwen2.5-0.5B-Instruct"' in script
    assert "--served-model-name Qwen2.5-0.5B-Instruct" in script


def test_tensor_parallel_matches_the_number_of_gpus():
    assert "--tensor-parallel-size 1 " in render(gpus=1)
    assert "--tensor-parallel-size 2 " in render(gpus=2)


def test_custom_all_reduce_is_disabled_permanently():
    # vllm#26480: TP em Ampere + cudagraph FULL + prefix caching gerava saída vazia
    assert "--disable-custom-all-reduce" in render(gpus=2)
    assert "--disable-custom-all-reduce" in render(gpus=1)


def test_dtype_and_memory_settings_reach_vllm():
    script = render(dtype="half")
    assert "--dtype half" in script
    assert "--gpu-memory-utilization 0.3" in script
    assert "--max-model-len 4096" in script


def test_trust_remote_code_is_passed_only_when_asked():
    assert "--trust-remote-code" not in render()
    assert "--gpu-memory-utilization 0.3 --disable-custom-all-reduce &" in render()
    script = render(trust_remote_code=True)
    assert "--gpu-memory-utilization 0.3 --disable-custom-all-reduce --trust-remote-code &" in script


def test_model_is_loaded_offline_from_the_local_folder():
    assert "export HF_HUB_OFFLINE=1" in render()


def test_harmony_vocab_cache_is_pointed_at_the_prefetched_dir():
    # gpt-oss: sem isso o tokenizer tenta baixar o vocab do nó de computação
    assert 'export TIKTOKEN_RS_CACHE_DIR="$ROOT/tiktoken-cache"' in render()


def test_token_is_generated_on_the_node_not_embedded():
    script = render()
    assert "secrets.token_hex(24)" in script
    assert "VLLM_API_KEY" in script
    assert "--api-key" not in script


def test_state_file_is_written_with_restrictive_umask():
    script = render()
    assert "umask 077" in script
    assert "write_state ready" in script


def test_validate_model_accepts_hf_repo_ids():
    assert validate_model("Qwen/Qwen2.5-0.5B-Instruct") == "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.mark.parametrize("bad", ["../etc", "Qwen", "a/b c", "a/b;rm -rf ~", "a/.."])
def test_validate_model_rejects_anything_unsafe_for_the_shell(bad):
    with pytest.raises(ValueError):
        validate_model(bad)

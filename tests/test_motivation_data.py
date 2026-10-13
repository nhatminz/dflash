import pandas as pd
import pytest

from analysis.verification_feedback.config import Config
from analysis.verification_feedback.data import prepare_plan, question_hash, read_questions
from analysis.verification_feedback.io import read_json


class Tokenizer:
    chat_template = "fixture"
    eos_token_id = 9

    def apply_chat_template(self, messages, **kwargs):
        return str(messages)

    def encode(self, text, **kwargs):
        return list(range(len(text)))


def fixture(tmp_path, monkeypatch, test_size=160):
    import transformers

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: Tokenizer())
    train, test = tmp_path / "train.parquet", tmp_path / "test.parquet"
    pd.DataFrame(dict(question=[f"training question {i}" for i in range(8)])).to_parquet(train)
    pd.DataFrame(dict(question=[f"held-out question {i}" for i in range(test_size)])).to_parquet(
        test
    )
    config = Config(run_dir=tmp_path / "run", output_dir=tmp_path / "analysis", test_data=str(test))
    record = dict(
        recipe=dict(
            policy=dict(
                tokenizer={"name": "fixture", "chat_template": "default"},
                draft={"gamma": 15},
                generation=dict(
                    stop_token_ids=None, max_new_tokens=2048, vllm_cfg={"max_model_len": 4112}
                ),
            ),
            data=dict(train={"data_path": str(train)}, max_input_seq_length=2048),
        ),
        recipe_sha256="fixture",
        prompt_source_sha256="fixture",
        templates=dict(SYSTEM_PROMPT="system", USER_PROMPT="{instruction}"),
    )
    return config, record


def test_same_160_distinct_prompts_and_deterministic_seeds_at_both_stages(tmp_path, monkeypatch):
    config, record = fixture(tmp_path, monkeypatch)
    a = prepare_plan(config, record)
    b = prepare_plan(config, record)
    assert a == b
    hashes = [{r["question_sha256"] for r in rollout["requests"]} for rollout in a["rollouts"]]
    assert len(set.union(*hashes)) == 160
    assert all(len(group) == 8 for group in hashes)
    assert sum(len(r["requests"]) for r in a["rollouts"]) == 1280
    assert a["generation"]["max_new_tokens"] == 2048
    assert read_json(config.destination / "evaluation_manifest.json") == a


def test_train_test_hash_leakage_is_fatal(tmp_path, monkeypatch):
    config, record = fixture(tmp_path, monkeypatch)
    pd.DataFrame(dict(question=["training question 0"])).to_parquet(config.test_data)
    with pytest.raises(ValueError, match="leakage"):
        prepare_plan(config, record)


def test_missing_or_insufficient_test_split_never_falls_back_to_train(tmp_path, monkeypatch):
    config, record = fixture(tmp_path, monkeypatch, test_size=159)
    with pytest.raises(ValueError, match="160 required"):
        prepare_plan(config, record)
    from pathlib import Path

    Path(config.test_data).unlink()
    with pytest.raises(FileNotFoundError):
        prepare_plan(config, record)


def test_normalized_question_dedup_and_split_metadata(tmp_path):
    assert question_hash(" Question:  a   b\nAnswer: Let's think step by step.") == question_hash(
        "a b"
    )
    p = tmp_path / "test.parquet"
    pd.DataFrame(dict(question=["a  b", "a b"])).to_parquet(p)
    rows, _ = read_questions(p, "test")
    assert len(rows) == 1
    pd.DataFrame([dict(question="q", extra_info={"split": "train"})]).to_parquet(p)
    with pytest.raises(ValueError, match="Split metadata"):
        read_questions(p, "test")

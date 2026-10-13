"""Official held-out SimpleLR selection and the recorded training prompt format."""

import hashlib
import random
import re
import unicodedata
from pathlib import Path

from .io import digest, file_hash, immutable_json


def question_hash(text):
    text = unicodedata.normalize("NFKC", text).strip()
    text = re.sub(r"^Question:\s*", "", text)
    text = re.sub(r"\s*Answer:\s*Let's think step by step\.\s*$", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return hashlib.sha256(text.encode()).hexdigest()


def read_questions(path, split):
    import pandas as pd

    path = Path(path).resolve(strict=True)
    if path.suffix != ".parquet":
        raise ValueError(f"Expected an explicit SimpleLR {split} Parquet split: {path}")
    rows, seen = [], set()
    for index, item in enumerate(pd.read_parquet(path).to_dict("records")):
        extra = item.get("extra_info")
        if isinstance(extra, dict) and extra.get("split", split) != split:
            raise ValueError(f"Split metadata disagrees with {split}: row {index}")
        source = item.get("data_source")
        if source is not None and source not in {
            "simplelr",
            "simplelr_abel",
            "simplelr_abel_level3to5",
        }:
            raise ValueError(f"Not a SimpleLR row: {source}")
        question = item.get("question")
        if question is None or pd.isna(question):
            prompt = item.get("prompt")
            if isinstance(prompt, str):
                question = prompt
            else:
                # Match LocalMathDataset/load_questions, including whitespace.
                question = prompt[0]["content"]
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Invalid {split} question at row {index}")
        key = question_hash(question)
        if key not in seen:
            seen.add(key)
            rows.append(
                dict(prompt_id=key, question_sha256=key, question=question, source_row=index)
            )
    return rows, dict(path=str(path), sha256=file_hash(path), unique_questions=len(rows))


def prepare_plan(c, record):
    from transformers import AutoTokenizer

    recipe = record["recipe"]
    train_path = c.train_data or recipe["data"]["train"]["data_path"]
    recorded_train = Path(recipe["data"]["train"]["data_path"])
    if c.train_data and Path(c.train_data).resolve() != recorded_train.resolve():
        if not recorded_train.is_file() or file_hash(c.train_data) != file_hash(recorded_train):
            raise ValueError(
                "--train-data must be a byte-identical copy of the actual recorded training split"
            )
    if Path(train_path).resolve() == Path(c.test_data).resolve():
        raise ValueError("Train and test must be separate official split files")
    train, train_info = read_questions(train_path, "train")
    test, test_info = read_questions(c.test_data, "test")
    overlap = {r["question_sha256"] for r in train} & {r["question_sha256"] for r in test}
    if overlap:
        raise ValueError(
            f"Train/test question-hash leakage: {len(overlap)} questions; refusing held-out evaluation"
        )
    tokenizer_cfg = recipe["policy"]["tokenizer"]
    tokenizer_kwargs = dict(tokenizer_cfg.get("tokenizer_kwargs") or {})
    tokenizer_kwargs.update(local_files_only=True, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_cfg["name"], **tokenizer_kwargs)
    chat = tokenizer_cfg.get("chat_template", "default")
    if chat is None:
        raise ValueError("Passthrough chat formatting is not this run's specnaacl processor")
    if chat.lower() != "default":
        tokenizer.chat_template = Path(chat).read_text() if chat.endswith(".jinja") else chat
    required = c.eval_rollouts * c.prompts_per_rollout
    limit = int(recipe["data"]["max_input_seq_length"])
    retained, filtered = [], 0
    templates = record["templates"]
    for row in test:
        messages = [
            dict(role="system", content=templates["SYSTEM_PROMPT"]),
            dict(role="user", content=templates["USER_PROMPT"].format(instruction=row["question"])),
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **(tokenizer_cfg.get("chat_template_kwargs") or {}),
        )
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) >= limit:
            filtered += 1
            continue
        retained.append(
            dict(
                row, prompt_token_ids=ids, rendered_sha256=hashlib.sha256(text.encode()).hexdigest()
            )
        )
    if len(retained) < required:
        raise ValueError(
            f"Only {len(retained)} distinct held-out prompts survive original length filtering; "
            f"{required} required. Supply an explicit disjoint test split; no train fallback"
        )
    random.Random(c.seed).shuffle(retained)
    selected = retained[:required]
    g = recipe["policy"]["generation"]
    stop = g.get("stop_token_ids")
    if stop is None:
        stop = [tokenizer.eos_token_id]
    if g.get("ignore_eos", False):
        raise ValueError("This experiment expects original natural EOS termination")
    max_model_len = g["vllm_cfg"].get("max_model_len")
    if not isinstance(max_model_len, int):
        raise ValueError("Expanded source config must record generation.vllm_cfg.max_model_len")
    gamma = recipe["policy"]["draft"]["gamma"]
    rollouts = []
    for rollout in range(c.eval_rollouts):
        prompts = selected[rollout * c.prompts_per_rollout : (rollout + 1) * c.prompts_per_rollout]
        requests = []
        for p, row in enumerate(prompts):
            for sample in range(c.responses_per_prompt):
                response = p * c.responses_per_prompt + sample
                # Exactly the pinned NeMo VllmGenerationWorker's speculative
                # context clamp, preserving the original generation cap.
                cap = min(
                    int(g["max_new_tokens"]),
                    max(1, max_model_len - len(row["prompt_token_ids"]) - (gamma + 1)),
                )
                requests.append(
                    dict(
                        row,
                        response_id=response,
                        sample_index=sample,
                        stop_token_ids=sorted(set(stop + [tokenizer.eos_token_id])),
                        max_tokens=cap,
                        seed=int(digest([c.seed, rollout, response])[:8], 16),
                    )
                )
        rollouts.append(dict(rollout_id=rollout, requests=requests))
    plan = dict(
        schema="dflash_verification_feedback_v1",
        seed=c.seed,
        eval_rollouts=c.eval_rollouts,
        prompts_per_rollout=c.prompts_per_rollout,
        responses_per_prompt=c.responses_per_prompt,
        distinct_prompts=required,
        train=train_info,
        test=dict(test_info, overlength_filtered=filtered),
        overlap_question_hashes=0,
        tokenizer_name=tokenizer_cfg["name"],
        tokenizer_chat_template_sha256=hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        recipe_sha256=record["recipe_sha256"],
        prompt_source_sha256=record["prompt_source_sha256"],
        generation=g,
        rollouts=rollouts,
    )
    immutable_json(c.destination / "evaluation_manifest.json", plan)
    return plan

"""Local datasets and chat formatting, equivalent to SpecNaacl."""

import json
import math
import random
from pathlib import Path

import torch

SYSTEM_PROMPT = "You are a math problem assistant."
USER_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context.
            Write a response that appropriately completes the request.
            Your response should include your thought process enclosed within <think></think> tags
            and the final answer enclosed within <answer></answer> tags (Just put a number between the tags).\n
            ### Instruction:\n{instruction}\nPlease reason step by step, and put your final answer within \\boxed{{}}"""
PATHS = dict(
    gsm8k="gsm8k/main/train-00000-of-00001.parquet",
    simplelr="simplelr_abel_level3to5/train.parquet",
    dapo="DAPO-Math-17k-Processed/en/train-00000-of-00001.parquet",
)


def select_subset(rows, max_samples=0, seed=42, fraction=1.0):
    target = len(rows) if fraction >= 1 else max(1, math.ceil(len(rows) * fraction))
    if max_samples:
        target = min(target, max_samples)
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    return [rows[i] for i in sorted(indices[:target])]


def load_questions(path, dataset, max_samples=0, seed=42):
    if dataset not in PATHS:
        raise ValueError("DATASET must be gsm8k, simplelr, or dapo")
    import pandas as pd

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix == ".parquet":
        rows = pd.read_parquet(path).to_dict(orient="records")
    elif path.suffix in {".json", ".jsonl"}:
        rows = (
            [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            if path.suffix == ".jsonl"
            else json.loads(path.read_text())
        )
    else:
        raise ValueError(f"unsupported local dataset {path}")
    result = []
    for item in rows:
        question = item.get("question")
        if question is None or (isinstance(question, float) and math.isnan(question)):
            prompt = item["prompt"]
            question = prompt if isinstance(prompt, str) else prompt[0]["content"]
        answer = item.get("answer")
        if (answer is None or (isinstance(answer, float) and math.isnan(answer))) and isinstance(
            item.get("reward_model"), dict
        ):
            answer = item["reward_model"]["ground_truth"]
        if answer is None:
            raise ValueError("dataset row has no mathematical ground truth")
        answer = str(answer).split("####")[-1].strip()
        if "\\boxed{" not in answer:
            answer = f"\\boxed{{{answer}}}"
        if not isinstance(question, str):
            raise ValueError("question must be a string")
        result.append(dict(question=question, answer=answer))
    return select_subset(result, max_samples, seed)


class ShareGPTCollator:
    def __init__(self, tokenizer, max_length=2048):
        self.tokenizer, self.max_length = tokenizer, max_length

    def encode(self, example):
        tokenizer = self.tokenizer
        ids = tokenizer.encode(
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n", add_special_tokens=False
        )
        mask = [0] * len(ids)
        for turn in example["conversations"]:
            role = {"human": "user", "gpt": "assistant"}.get(turn["from"], turn["from"])
            tokens = tokenizer.encode(
                f"<|im_start|>{role}\n{turn['value']}<|im_end|>\n", add_special_tokens=False
            )
            ids.extend(tokens)
            mask.extend([int(role in ("assistant", "ASSISTANT"))] * len(tokens))
        return ids[: self.max_length], mask[: self.max_length]

    def __call__(self, examples):
        rows = [self.encode(example) for example in examples]
        size = max(len(ids) for ids, _ in rows)
        return dict(
            input_ids=torch.tensor(
                [ids + [self.tokenizer.eos_token_id] * (size - len(ids)) for ids, _ in rows]
            ),
            attention_mask=torch.tensor(
                [[1] * len(ids) + [0] * (size - len(ids)) for ids, _ in rows]
            ),
            loss_mask=torch.tensor([mask + [0] * (size - len(mask)) for _, mask in rows]),
        )


def load_sharegpt(path, collator, max_samples=0):
    rows = json.loads(Path(path).read_text())
    if max_samples:
        rows = rows[:max_samples]
    # Match assistant masks; eliminate all-prompt examples deterministically.
    valid = [row for row in rows if any(collator.encode(row)[1][1:])]
    if not valid:
        raise ValueError("no ShareGPT assistant tokens survive the max length filter")
    return valid

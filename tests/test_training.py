import json

import pytest
import torch

from dflash_reflex.data import ShareGPTCollator, load_questions, load_sharegpt, select_subset
from dflash_reflex.rewards import accuracy_reward_func, format_reward_func


@pytest.mark.parametrize("dataset", ["gsm8k", "simplelr", "dapo"])
def test_local_math_datasets(tmp_path, dataset):
    file = tmp_path / "rows.json"
    file.write_text(
        json.dumps(
            [
                dict(question="2+2?", answer="work #### 4"),
                dict(prompt=[dict(content="half?")], reward_model=dict(ground_truth="1/2")),
            ]
        )
    )
    rows = load_questions(file, dataset)
    assert rows[0]["answer"] == r"\boxed{4}"
    assert rows[1]["question"] == "half?"
    assert select_subset(rows, 1, 42) == select_subset(rows, 1, 42)


def test_math_equivalence_and_format():
    assert accuracy_reward_func([r"\boxed{0.5}"], [r"\boxed{\frac{1}{2}}"])[0] == 1
    assert accuracy_reward_func([r"\boxed{3}"], [r"\boxed{4}"])[0] == 0
    assert format_reward_func(["<think>work\n</think>\nanswer"])[0] == 1


def test_sharegpt_mask_and_padding(tmp_path):
    class Tokenizer:
        eos_token_id = 0

        def encode(self, text, **kwargs):
            return [ord(x) % 255 + 1 for x in text]

    collator = ShareGPTCollator(Tokenizer(), 512)
    rows = [
        dict(
            conversations=[
                dict(**{"from": "human", "value": "Hi"}),
                dict(**{"from": "gpt", "value": "Hello"}),
            ]
        )
    ]
    path = tmp_path / "sharegpt.json"
    path.write_text(json.dumps(rows))
    selected = load_sharegpt(path, collator)
    batch = collator(selected)
    assert batch["loss_mask"].sum() > 0
    assert not batch["loss_mask"][0, :20].any()
    assert batch["input_ids"].dtype == torch.long

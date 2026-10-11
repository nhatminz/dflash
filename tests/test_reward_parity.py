import importlib.util
import json
from pathlib import Path

import pytest

from dflash_reflex import rewards
from dflash_reflex.data import SYSTEM_PROMPT, USER_PROMPT, load_questions, select_subset

ROOT = Path(__file__).resolve().parents[1]


def load_spec(relative):
    spec = importlib.util.spec_from_file_location(
        "source_" + Path(relative).stem, ROOT.parent / "SpecNaacl" / relative
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_actual_reward_source_parity():
    original = load_spec("helper/rewards.py")
    answers = ["<think>work\n</think>\n\\boxed{4}", "\\boxed{5}", "no math", "\\boxed{1/2}"]
    gold = ["\\boxed{4}", "\\boxed{4}", "unparseable", "\\boxed{0.5}"]
    assert rewards.format_reward_func(answers) == original.format_reward_func(answers)
    assert rewards.accuracy_reward_func(answers, gold) == original.accuracy_reward_func(
        answers, gold
    )


@pytest.mark.parametrize("dataset", ["gsm8k", "simplelr", "dapo"])
def test_rows_subset_and_exact_prompt_strings(dataset, tmp_path):
    original = load_spec("helper/get_QAs.py")
    examples = [
        {"question": "q1", "answer": "reason #### 42"},
        {"prompt": [{"content": "q2"}], "reward_model": {"ground_truth": " 5 "}},
        {"question": "q3", "answer": "\\boxed{9}"},
    ]
    # Separate rows avoid pandas NaN coercion changing the legacy key precedence.
    got = []
    expected = []
    for i, item in enumerate(examples):
        path = tmp_path / f"{i}.json"
        path.write_text(json.dumps([item]))
        got.extend(load_questions(path, dataset))
        expected.extend(original.get_QAs_from_path(str(path), "train"))
    assert got == expected
    assert select_subset(got, 2, 19) == original.select_train_subset(got, max_samples=2, seed=19)
    assert SYSTEM_PROMPT == original.prompt_dict["math"]["system_prompt"]
    assert USER_PROMPT == original.prompt_dict["math"]["user_prompt"]

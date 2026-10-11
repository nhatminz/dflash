"""NeMo dataset extension, loaded through its documented dotted-class registry."""

from datasets import Dataset
from nemo_rl.data.datasets.raw_dataset import RawDataset
from nemo_rl.data.processors import register_processor

from .data import SYSTEM_PROMPT, USER_PROMPT, load_questions


def specnaacl_processor(datum, task_spec, tokenizer, max_seq_length, idx):
    messages = [
        dict(role="system", content=SYSTEM_PROMPT),
        dict(role="user", content=USER_PROMPT.format_map(dict(instruction=datum["question"]))),
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    if len(ids) >= max_seq_length:
        raise ValueError("overlength prompt escaped the dataset's deterministic length filter")
    return dict(
        message_log=[dict(role="user", content=text, token_ids=ids)],
        length=len(ids),
        extra_env_info=dict(ground_truth=datum["answer"]),
        loss_multiplier=1.0,
        idx=idx,
        task_name=datum["task_name"],
    )


register_processor("specnaacl", specnaacl_processor)


class LocalMathDataset(RawDataset):
    def __init__(
        self, data_path, dataset, model_path, max_prompt_length, max_samples=0, seed=42, **kwargs
    ):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        rows = load_questions(data_path, dataset, max_samples, seed)
        self.task_name = dataset
        retained = []
        for row in rows:
            messages = [
                dict(role="system", content=SYSTEM_PROMPT),
                dict(
                    role="user", content=USER_PROMPT.format_map(dict(instruction=row["question"]))
                ),
            ]
            ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            if len(ids) < max_prompt_length:
                retained.append(dict(**row, task_name=dataset))
        if not retained:
            raise ValueError("all prompts were filtered by MAX_PROMPT_LENGTH")
        self.dataset = Dataset.from_list(retained)
        self.val_dataset = None

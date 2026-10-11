"""Validate and optionally export the exact local math data selection."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from dflash_reflex.data import PATHS, SYSTEM_PROMPT, USER_PROMPT, load_questions

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=PATHS, default="simplelr")
    p.add_argument("--data-root", default="/workspace/storage-shared/nlp/minhpn19/data")
    p.add_argument("--path")
    p.add_argument("--model", help="Also run the actual chat-template length filter")
    p.add_argument("--max-prompt-length", type=int, default=2048)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output")
    a = p.parse_args()
    path = a.path or str(Path(a.data_root) / PATHS[a.dataset])
    rows = load_questions(path, a.dataset, a.max_samples, a.seed)
    before = len(rows)
    if a.model:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
        rows = [
            r
            for r in rows
            if len(
                tokenizer.apply_chat_template(
                    [
                        dict(role="system", content=SYSTEM_PROMPT),
                        dict(
                            role="user",
                            content=USER_PROMPT.format_map(dict(instruction=r["question"])),
                        ),
                    ],
                    tokenize=True,
                    add_generation_prompt=True,
                )
            )
            < a.max_prompt_length
        ]
    if not rows:
        raise ValueError("no valid training prompts")
    if a.output:
        Path(a.output).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(
        json.dumps(
            dict(dataset=a.dataset, source=path, selected=before, retained=len(rows), seed=a.seed),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

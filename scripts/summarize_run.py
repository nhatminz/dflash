import argparse
import csv
import json
from pathlib import Path


def summarize(path):
    with (path / "logs/rollout_timing.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("run has no committed iteration rows")
    sums = {
        name: sum(float(row[name]) for row in rows)
        for name in (
            "iter_generated_tokens",
            "iter_accepted_tokens",
            "iter_verification_rounds",
            "iter_accepted_draft_tokens",
            "iter_proposed_draft_tokens",
            "iter_generation_time_s",
            "iter_wall_time_s",
        )
    }

    def ratio(a, b):
        return sums[a] / sums[b] if sums[b] else 0.0

    return dict(
        run=str(path),
        iterations=len(rows),
        method=rows[-1]["method"],
        dataset=rows[-1]["dataset"],
        aal=ratio("iter_accepted_tokens", "iter_verification_rounds"),
        acceptance_rate=ratio("iter_accepted_draft_tokens", "iter_proposed_draft_tokens"),
        generation_tokens_per_s=ratio("iter_generated_tokens", "iter_generation_time_s"),
        end_to_end_tokens_per_s=ratio("iter_generated_tokens", "iter_wall_time_s"),
        total_iteration_wall_time_s=sums["iter_wall_time_s"],
        total_training_time_s=float(rows[-1]["cumulative_wall_time_s"]),
        **sums,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps([summarize(path) for path in args.runs], indent=2))

import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", default="training_time.png")
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    for run in args.runs:
        with (run / "logs/rollout_timing.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        ax.plot(
            [int(r["iteration"]) for r in rows],
            [float(r["cumulative_wall_time_s"]) / 60 for r in rows],
            label=run.name,
        )
    ax.set(xlabel="GRPO iteration", ylabel="Cumulative training wall time (minutes)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize="small")
    fig.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()

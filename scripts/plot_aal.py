import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", default="aal.png")
    parser.add_argument("--cumulative", action="store_true")
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    for run in args.runs:
        with (run / "logs/rollout_timing.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        key = "cumulative_aal" if args.cumulative else "iter_aal"
        valid = [r for r in rows if r[key] not in ("", "null", "None")]
        if valid:
            ax.plot(
                [int(r["iteration"]) for r in valid], [float(r[key]) for r in valid], label=run.name
            )
        else:
            print(f"{run.name}: AAL unavailable (production baseline); use instrumented comparison")
    ax.set(
        xlabel="GRPO iteration",
        ylabel="Accepted path tokens / verification",
        title="DFlash average accepted length",
    )
    ax.grid(alpha=0.25)
    if ax.lines:
        ax.legend(fontsize="small")
    fig.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()

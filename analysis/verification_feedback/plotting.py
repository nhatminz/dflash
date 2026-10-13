"""Publication figures from measured statistics only, including unavailable support."""

from pathlib import Path


def draw_panels(axes, results):
    colors = ("#2369A1", "#D46A22", "#31825A", "#9763A2")
    from .statistics import DISTANCES

    any_a = any_b = any_c = False
    steps = sorted(results, key=int)
    for index, step in enumerate(steps):
        result = results[step]
        color = colors[index % len(colors)]
        label = f"Step {step}"
        for x, row in enumerate(result["fig1a"]["distances"]):
            stat = row["matched"]
            value = stat["risk_ratio"]
            if value is not None:
                any_a = True
                point(
                    axes[0],
                    x + (index - (len(steps) - 1) / 2) * 0.08,
                    value,
                    stat["risk_ratio_ci95"],
                    color,
                    label
                    if not any(
                        v["matched"]["risk_ratio"] is not None
                        for v in result["fig1a"]["distances"][:x]
                    )
                    else None,
                )
        control = result["fig1a"]["previous_rollout_control"]["matched"]
        if control["risk_ratio"] is not None:
            any_a = True
            axes[0].axhline(
                control["risk_ratio"],
                color=color,
                linestyle="--",
                linewidth=1,
                label=f"{label}: independent rollout",
            )
            lo, hi = control["risk_ratio_ci95"]
            if lo is not None and hi is not None:
                axes[0].axhspan(lo, hi, color=color, alpha=0.06)
        stat = result["fig1b"]["exclusive_sibling_vs_none"]["matched"]
        if stat["risk_ratio"] is not None:
            any_b = True
            point(axes[1], index, stat["risk_ratio"], stat["risk_ratio_ci95"], color, label)
        for scope, key in enumerate(
            ("within_response_prior_miss_share", "within_prompt_prior_miss_share")
        ):
            stat = result["fig1c"][key]
            if stat["estimate"] is not None:
                any_c = True
                width = 0.7 / max(1, len(steps))
                x = scope + (index - (len(steps) - 1) / 2) * width
                axes[2].bar(
                    x,
                    100 * stat["estimate"],
                    width=width * 0.92,
                    color=color,
                    alpha=0.9,
                    label=label if scope == 0 else None,
                )
                point(
                    axes[2],
                    x,
                    100 * stat["estimate"],
                    [100 * v if v is not None else None for v in stat["ci95"]],
                    color,
                    None,
                    marker="none",
                )
    axes[0].set_xticks(range(len(DISTANCES)), DISTANCES)
    axes[0].set_xlabel("Verification-round distance (same response)")
    axes[0].set_ylabel("Matched miss risk ratio")
    axes[0].set_title("(a) Recurring misses")
    axes[1].set_xticks(range(len(steps)), [f"Step {s}" for s in steps])
    axes[1].set_ylabel("Matched miss risk ratio")
    axes[1].set_title("(b) Same-prompt sibling feedback")
    axes[2].set_xticks([0, 1], ["Same response", "Same prompt\n(inclusive)"])
    axes[2].set_ylabel("Misses with earlier feedback (%)")
    axes[2].set_ylim(0, 105)
    axes[2].set_title("(c) Feedback already available")
    for axis, available in zip(axes, (any_a, any_b, any_c)):
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
        if not available:
            axis.text(
                0.5,
                0.55,
                "Insufficient statistical support",
                transform=axis.transAxes,
                ha="center",
                va="center",
                fontsize=8,
            )
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(handles, labels, frameon=False, fontsize=7)
    for axis in axes[:2]:
        axis.axhline(1, color="#777777", linestyle=":", linewidth=0.8)


def point(axis, x, value, interval, color, label, marker="o"):
    axis.plot([x], [value], marker=marker, color=color, label=label)
    lo, hi = interval
    if lo is not None and hi is not None:
        # Draw actual percentile bounds; do not clamp a CI around its estimate.
        axis.vlines(x, lo, hi, color=color, linewidth=1)
        axis.hlines([lo, hi], x - 0.025, x + 0.025, color=color, linewidth=1)


def figures(root, results):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(root)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.4), layout="constrained")
    draw_panels(axes, results)
    fig.savefig(root / "fig1_dflash_combined.pdf")
    fig.savefig(root / "fig1_dflash_combined.png", dpi=300)
    plt.close(fig)
    for index, name in enumerate(("a", "b", "c")):
        fig, all_axes = plt.subplots(1, 3, figsize=(13.5, 3.4), layout="constrained")
        draw_panels(all_axes, results)
        fig.canvas.draw()
        extent = (
            all_axes[index]
            .get_tightbbox(fig.canvas.get_renderer())
            .transformed(fig.dpi_scale_trans.inverted())
        )
        fig.savefig(root / f"fig1{name}_dflash.pdf", bbox_inches=extent.expanded(1.04, 1.08))
        plt.close(fig)

"""Plot aggregate sim-to-sim validation statistics.

Each bar chart overlays the convergence count (wide, translucent bar)
and task-success count (narrow, opaque bar). Starred drift cases are
included as passes, consistently with the validation tables. Every
configuration has 5 seeds x 8 tasks = 40 trials.

Fill RESULTS below with your own (success, convergence) counts from
``main2.m``. Leave a tuple as ``(None, None)`` to omit that bar.
This file does not ship paper numbers.

Replace --output-dir with a local folder for the PNG/PDF figures.

Example::

    python scripts/plot_validation.py --output-dir path/to/output_figures
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


MAX_TRIALS = 40
Result = Tuple[Optional[int], Optional[int]]  # (success, convergence)

# Replace (None, None) with (task_success, convergence) from your main2.m runs.
# Each entry is out of MAX_TRIALS. Omit a bar by leaving it as (None, None).
RESULTS = {
    "comparison": {
        "Predictor\n$\\lambda=0.25,\\ \\beta=1$": (None, None),
        "Vanilla SAC": (None, None),
        "Domain\nRandomization": (None, None),
    },
    "lambda": {
        "$\\lambda=0.10$": (None, None),
        "$\\lambda=0.25$": (None, None),
        "$\\lambda=0.50$": (None, None),
    },
    "beta": {
        "$\\beta=1$": (None, None),
        "$\\beta=2$": (None, None),
        "$\\beta=4$": (None, None),
    },
    "batch": {
        "$B=64$": (None, None),
        "$B=128$": (None, None),
        "$B=256$": (None, None),
    },
    "ablation": {
        "Predictor\n$\\lambda=0.25,\\ \\beta=1$": (None, None),
        "Predictor\n$\\lambda=0,\\ \\beta=1$": (None, None),
        "Predictor\n$\\lambda=0.25,\\ \\beta=0$": (None, None),
    },
}


def _validate_results(results: Mapping[str, Result]) -> None:
    """Reject malformed aggregate counts before plotting."""
    for label, (success, convergence) in results.items():
        for metric_name, value in (
            ("success", success),
            ("convergence", convergence),
        ):
            if value is not None and not 0 <= value <= MAX_TRIALS:
                raise ValueError(
                    f"{label}: {metric_name}={value} is outside "
                    f"[0, {MAX_TRIALS}]."
                )


def plot_overlay(
    results: Mapping[str, Result],
    title: str,
    output_stem: Path,
    *,
    dpi: int,
    show: bool,
) -> None:
    """Create one overlaid success/convergence bar chart."""
    _validate_results(results)

    missing = [
        label
        for label, (success, convergence) in results.items()
        if success is None or convergence is None
    ]
    if missing:
        print(
            f"[{output_stem.name}] omitted (missing statistics): "
            + ", ".join(label.replace("\n", " ") for label in missing)
        )

    available = [
        (label, success, convergence)
        for label, (success, convergence) in results.items()
        if success is not None and convergence is not None
    ]
    if not available:
        raise ValueError(f"No complete statistics available for '{title}'.")

    labels = [row[0] for row in available]
    success = np.asarray([row[1] for row in available], dtype=float)
    convergence = np.asarray([row[2] for row in available], dtype=float)
    x = np.arange(len(labels))

    width = max(6.4, 1.75 * len(labels))
    fig, ax = plt.subplots(figsize=(width, 5.2))

    convergence_bars = ax.bar(
        x,
        convergence,
        width=0.72,
        color="#0072B2",
        alpha=0.32,
        edgecolor="#005A8D",
        linewidth=1.2,
        hatch="//",
        label="Convergence",
        zorder=2,
    )
    success_bars = ax.bar(
        x,
        success,
        width=0.44,
        color="#009E73",
        alpha=0.92,
        edgecolor="#006B4F",
        linewidth=1.0,
        label="Task success",
        zorder=3,
    )

    ax.bar_label(
        convergence_bars,
        labels=[str(int(value)) for value in convergence],
        padding=3,
        fontsize=10,
        color="#005A8D",
    )
    ax.bar_label(
        success_bars,
        labels=[str(int(value)) for value in success],
        padding=3,
        fontsize=10,
        fontweight="bold",
        color="#006B4F",
    )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Number of passed trials (out of 40)")
    ax.set_ylim(0, MAX_TRIALS)
    ax.set_yticks(np.arange(0, MAX_TRIALS + 1, 5))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.4, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved {output_stem.with_suffix('.png')}")
    print(f"Saved {output_stem.with_suffix('.pdf')}")

    if show:
        plt.show()
    plt.close(fig)


def generate_figure_set(
    data: Mapping[str, Mapping[str, Result]],
    output_dir: Path,
    *,
    dpi: int,
    show: bool,
) -> None:
    """Generate comparison, sensitivity, and ablation figures."""
    specifications: Sequence[Tuple[str, str]] = (
        (
            "comparison",
            "Performance Comparison ($B=256$)",
        ),
        (
            "lambda",
            r"Sensitivity to $\lambda$ ($\beta=1$, $B=256$)",
        ),
        (
            "beta",
            r"Sensitivity to $\beta$ ($\lambda=0.25$, $B=256$)",
        ),
        (
            "batch",
            r"Sensitivity to Batch Size ($\lambda=0.25$, $\beta=1$)",
        ),
        (
            "ablation",
            "Ablation Study ($B=256$)",
        ),
    )

    for key, title in specifications:
        results = data.get(key, {})
        complete = any(
            success is not None and convergence is not None
            for success, convergence in results.values()
        )
        if not complete:
            print(
                f"Skip {key}: fill (success, convergence) "
                "in this script before plotting."
            )
            continue
        plot_overlay(
            results,
            title,
            output_dir / key,
            dpi=dpi,
            show=show,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate overlaid task-success and convergence bar charts "
            "for sim-to-sim validation."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "validation_figures",
        help="Directory for PNG and PDF figures. Replace with your local path.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display each figure in addition to saving it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.titlelocation": "center",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    generate_figure_set(
        RESULTS,
        args.output_dir,
        dpi=args.dpi,
        show=args.show,
    )

    print(
        f"Finished. Figures (if any counts were filled) are in "
        f"{args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()



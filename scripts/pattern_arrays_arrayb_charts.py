"""Charts for the Array B Stage 1 report.

Run: uv run --with matplotlib python scripts/pattern_arrays_arrayb_charts.py

Palette matches the Array A charts (dataviz six checks): blue #2a78d6, orange #eb6834, grays.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "text.color": INK,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "font.size": 10,
        "axes.titlesize": 12,
    }
)


def lift_heatmap(cells: pd.DataFrame) -> None:
    """Where on the grid does the model rank best? Top-decile lift over each cell's base rate."""
    g = cells.pivot(index="x_atr", columns="y_atr", values="top_decile_lift").sort_index()
    g = g[sorted(g.columns)]
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    im = ax.imshow(g.to_numpy(), cmap="Blues", aspect="auto", vmin=1.0)
    ax.set_xticks(range(len(g.columns)), [f"{c:g}" for c in g.columns])
    ax.set_yticks(range(len(g.index)), [f"{i:g}" for i in g.index])
    ax.set_xlabel("heat allowed before the idea is dead (ATR beyond the sweep)")
    ax.set_ylabel("come back by (ATR inside the level)")
    ax.set_title("Top-decile lift over each cell's own base rate")
    ax.grid(False)
    base = cells.pivot(index="x_atr", columns="y_atr", values="base_rate").sort_index()
    base = base[sorted(base.columns)]
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            v = g.to_numpy()[i, j]
            ax.text(
                j,
                i,
                f"{v:.2f}\n({base.to_numpy()[i, j] * 100:.0f}%)",
                ha="center",
                va="center",
                fontsize=8,
                color=INK if v < 1.8 else SURFACE,
            )
    fig.colorbar(im, ax=ax, label="lift")
    fig.text(
        0.01,
        0.01,
        "Base rate of each cell in brackets - the number the lift is measured against.",
        fontsize=8,
        color=INK2,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "arrayB_lift_grid.png", dpi=150)
    plt.close(fig)


def decile_chart(oos: pd.DataFrame, x: float, y: float) -> None:
    """Hit rate by model score decile for one cell, against that cell's base rate."""
    c = oos[(oos["x_atr"] == x) & (oos["y_atr"] == y)]
    order = np.argsort(c["p"].to_numpy(), kind="mergesort")
    bins = np.array_split(order, 10)
    yy = c["y"].to_numpy()
    rates = [yy[b].mean() * 100 for b in bins]
    base = yy.mean() * 100
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.bar(range(1, 11), rates, color=BLUE, width=0.72)
    ax.axhline(base, color=ORANGE, linewidth=1.6, linestyle="--")
    ax.text(10.5, base + 1.5, f"base rate {base:.0f}%", color=ORANGE, ha="right", fontsize=9)
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("model score decile (10 = highest scored)")
    ax.set_ylabel("came back before taking the heat (%)")
    ax.set_title(f"Array B out of sample: come back {x:g} ATR before {y:g} ATR of heat")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(OUT / "arrayB_deciles.png", dpi=150)
    plt.close(fig)


def windows_chart(pw: pd.DataFrame) -> None:
    done = pw[~pw["skipped"].astype(bool)]
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    ax.bar(range(len(done)), done["brier_skill"] * 100, color=BLUE, width=0.7)
    ax.axhline(0, color=INK2, linewidth=1.0)
    ax.set_xticks(range(len(done)), [str(t)[:7] for t in done["test_start"]], rotation=45, ha="right")
    ax.set_ylabel("Brier skill vs the per-cell base rate (%)")
    ax.set_title("Every out-of-sample window, Array B")
    fig.tight_layout()
    fig.savefig(OUT / "arrayB_windows.png", dpi=150)
    plt.close(fig)


def main() -> int:
    cells = pd.read_csv(OUT / "arrayB_cells_oos.csv")
    pw = pd.read_csv(OUT / "arrayB_per_window.csv")
    oos = pd.read_parquet(OUT / "arrayB_oos_scores.parquet")
    summary = json.loads((OUT / "arrayB_summary.json").read_text())
    lift_heatmap(cells)
    decile_chart(oos, 0.50, 1.00)
    windows_chart(pw)
    print("wrote arrayB_lift_grid.png, arrayB_deciles.png, arrayB_windows.png")
    print(f"pooled lift {summary['pooled']['mean_cell_lift']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

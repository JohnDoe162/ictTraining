"""Charts for the Array A Stage 1 report. Run: uv run --with matplotlib python scripts/pattern_arrays_charts.py

Palette validated (dataviz six checks): blue #2a78d6, orange #eb6834, neutral grays.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

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
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def pct(x):
    return f"{100 * x:.0f}%"


def chart_reliability():
    rel = pd.read_csv(OUT / "arrayA_reliability.csv")
    fig, ax = plt.subplots(figsize=(7, 6), dpi=200)
    ax.plot([0, 1], [0, 1], color=INK2, lw=1.2, ls="--", zorder=1)
    ax.annotate("perfect calibration", xy=(0.74, 0.70), color=INK2, fontsize=10, rotation=38)
    size = 60 + 240 * rel["n"] / rel["n"].max()
    ax.scatter(rel["mean_pred"], rel["obs_rate"], s=size, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.plot(rel["mean_pred"], rel["obs_rate"], color=BLUE, lw=2, zorder=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Model score (predicted probability the draw is reached)")
    ax.set_ylabel("Share of episodes where it WAS reached")
    ax.set_title(
        "The score means what it says\nPooled out-of-sample reliability, MGC 1h draw-on-liquidity, 2019-07..2025-10",
        loc="left",
        fontsize=12,
        color=INK,
    )
    ax.annotate("dot size = number of episodes in the bin", xy=(0.02, 0.93), fontsize=9, color=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "arrayA_reliability.png")
    plt.close(fig)


def chart_deciles():
    dec = pd.read_csv(OUT / "arrayA_decile.csv")
    oos = pd.read_parquet(OUT / "arrayA_oos_scores.parquet")
    base = oos["y"].mean()
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=200)
    bars = ax.bar(dec["decile"], dec["event_rate"], width=0.72, color=BLUE, zorder=3)
    for b in bars:
        b.set_linewidth(2)
        b.set_edgecolor(SURFACE)
    ax.axhline(base, color=INK2, lw=1.4, ls="--", zorder=4)
    ax.annotate(f"base rate {pct(base)}", xy=(0.6, base + 0.015), color=INK2, fontsize=10)
    for i in (0, 9):
        ax.annotate(
            pct(dec["event_rate"].iloc[i]),
            xy=(dec["decile"].iloc[i], dec["event_rate"].iloc[i] + 0.015),
            ha="center",
            fontsize=10,
            color=INK,
        )
    ax.set_xticks(dec["decile"])
    ax.set_xlabel("Model-score decile (1 = lowest scores, 10 = highest)")
    ax.set_ylabel("Share of episodes where the draw was reached")
    ax.set_ylim(0, 1)
    ax.set_title(
        "Higher score, higher hit rate - out of sample\nPooled OOS event rate by score decile vs the base rate",
        loc="left",
        fontsize=12,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(OUT / "arrayA_deciles.png")
    plt.close(fig)


def chart_windows():
    pw = pd.read_csv(OUT / "arrayA_per_window.csv", parse_dates=["test_start"])
    pw = pw[~pw["skipped"]]
    x = pw["test_start"].dt.strftime("%Y-%m")
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=200)
    ax.plot(x, pw["top_decile_rate"], color=BLUE, lw=2, marker="o", ms=7, label="Top-decile episodes")
    ax.plot(x, pw["base_rate"], color=ORANGE, lw=2, marker="o", ms=7, label="All episodes (base rate)")
    ax.annotate("top decile", xy=(len(x) - 1.1, pw["top_decile_rate"].iloc[-1] + 0.03), color=BLUE, ha="right")
    ax.annotate("base rate", xy=(len(x) - 1.1, pw["base_rate"].iloc[-1] - 0.05), color=ORANGE, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of episodes where the draw was reached")
    ax.set_xlabel("Walk-forward test window (start month, 6-month windows)")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(frameon=False, loc="lower left")
    ax.set_title(
        "The edge holds in every unseen window\nTop-decile vs base event rate per walk-forward test window (each trained on the prior 18 months)",
        loc="left",
        fontsize=12,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(OUT / "arrayA_windows.png")
    plt.close(fig)


if __name__ == "__main__":
    chart_reliability()
    chart_deciles()
    chart_windows()
    print("charts written to", OUT)

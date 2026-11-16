"""Array B (5m sweep and reversal) Stage 1: build, audit, gate, train walk-forward, report.

Run:  python scripts/pattern_arrays_arrayb_stage1.py --bars BARS [--no-entry CAL] [--rebuild]

BARS is a Parquet or CSV of 1-minute bars; see src/pattern_arrays/data.py for the required
columns. No price data ships with this repository.

Specification: METHOD.md, "Array B". The verdict thresholds in thresholds.json were registered
in writing BEFORE this script was first run, and are applied here. The episode-count gate runs
BEFORE training: below it, the script stops and reports rather than training on too little.

This is a concept model: no orders, no costs, so nothing here is a profit-and-loss result.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pattern_arrays import arrayb, walkforward_cond
from pattern_arrays.data import load_bars, load_no_entry

START = "2018-01-01"
END = "2025-10-17"  # the seal; the holdout is not read
OUT = ROOT / "results"
EPS_PQ = OUT / "arrayB_episodes.parquet"
PARAMS = {
    "array": "B_sweep_reversal_5m",
    "model": "logit_l2",
    "C": 1.0,
    "floor": "pivot_k1_5m",
    "features": "SPEC_v2_B_18plus2",
    "grid_x": list(arrayb.X_GRID),
    "grid_y": list(arrayb.Y_GRID),
    "train_months": 18,
    "test_months": 6,
}


def build(bars_path, no_entry_path) -> tuple[pd.DataFrame, dict]:
    df1m = load_bars(bars_path, START, END)
    return arrayb.build_rows(df1m, load_no_entry(no_entry_path))


def audit(episodes: pd.DataFrame) -> pd.DataFrame:
    """Independent episodes per year - one per swept level. Rows are 30x this and are NOT
    independent, so the gate is quoted here, in episodes (SPEC B.10.2)."""
    e = episodes.copy()
    e["year"] = e["t_dec"].dt.year
    tab = e.groupby("year").agg(
        episodes=("cand_id", "count"),
        high_side=("side", lambda s: int((s > 0).sum())),
        low_side=("side", lambda s: int((s < 0).sum())),
        median_sweep_depth_atr=("sweep_depth_atr", "median"),
        median_mfe_back_atr=("mfe_back_atr", "median"),
        median_mae_beyond_atr=("mae_beyond_atr", "median"),
    )
    return tab


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars", required=True, help="Parquet or CSV of 1-minute bars")
    ap.add_argument("--no-entry", default=None, help="optional no-entry session calendar (CSV)")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--audit-only", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    thr = json.loads((ROOT / "thresholds.json").read_text())["array_b"]

    if EPS_PQ.exists() and not args.rebuild:
        episodes = pd.read_parquet(EPS_PQ)
        counts = json.loads((OUT / "arrayB_counts.json").read_text())
    else:
        episodes, counts = build(args.bars, args.no_entry)
        episodes.to_parquet(EPS_PQ, index=False)
        (OUT / "arrayB_counts.json").write_text(json.dumps(counts, indent=2))
    print("counts:", json.dumps(counts, indent=2))

    tab = audit(episodes)
    tab.to_csv(OUT / "arrayB_audit_by_year.csv")
    print("\nEpisodes per year (the unit of independence):")
    print(tab.to_string())

    # --- the gate, BEFORE any training (SPEC B.8 G) ----------------------------------------
    total = len(episodes)
    worst_year = int(tab["episodes"].min()) if len(tab) else 0
    gate = {
        "episodes_total": total,
        "min_episodes_total": thr["min_episodes_total"],
        "worst_year": worst_year,
        "min_episodes_per_year": thr["min_episodes_per_year"],
        "pass": total >= thr["min_episodes_total"] and worst_year >= thr["min_episodes_per_year"],
    }
    (OUT / "arrayB_gate.json").write_text(json.dumps(gate, indent=2))
    print("\ngate:", json.dumps(gate, indent=2))
    if not gate["pass"]:
        print("\nSTOP: episode count is below the pre-registered minimum. Not training.")
        return 1
    if args.audit_only:
        return 0

    res = walkforward_cond.run(episodes, START, END)
    if res["pooled"] is None:
        print("\nSTOP: no walk-forward window had enough episodes.")
        return 1
    res["oos"].to_parquet(OUT / "arrayB_oos_scores.parquet", index=False)
    pw = pd.DataFrame([{k: v for k, v in w.items() if k != "coef"} for w in res["per_window"]])
    pw.to_csv(OUT / "arrayB_per_window.csv", index=False)
    res["cells"].to_csv(OUT / "arrayB_cells_oos.csv", index=False)
    res["reliability"].to_csv(OUT / "arrayB_reliability.csv", index=False)

    v = walkforward_cond.verdict(res["pooled"], thr)
    summary = {
        "pooled": res["pooled"],
        "verdict": v,
        "thresholds": thr,
        "gate": gate,
        "windows": len([w for w in res["per_window"] if not w.get("skipped")]),
        "windows_skipped": len([w for w in res["per_window"] if w.get("skipped")]),
        "coefs_last_window": next((w["coef"] for w in reversed(res["per_window"]) if not w.get("skipped")), None),
        "features": ["intercept"] + arrayb.FEATURES,
        "grid": res["grid"],
        "data": {"start": START, "end": END, "built": str(date.today())},
    }
    (OUT / "arrayB_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("\npooled:", json.dumps(res["pooled"], indent=2))
    print("\nper window:")
    print(pw.to_string())
    print("\nper cell (OOS):")
    print(res["cells"].to_string())
    print("\nverdict:", json.dumps(v, indent=2, default=str))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

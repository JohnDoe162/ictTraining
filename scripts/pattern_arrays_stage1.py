"""Array A (1h DOL) Stage 1: build the labelled dataset, audit it, train walk-forward, report.

Run:  python scripts/pattern_arrays_stage1.py --bars BARS [--no-entry CAL] [--rebuild]

BARS is a Parquet or CSV of 1-minute bars; see src/pattern_arrays/data.py for the required
columns. No price data ships with this repository.

This is a concept model: it places no orders and models no costs, so nothing here is a
profit-and-loss result. Verdict thresholds are NOT applied, because none were pre-registered
before the run; the output is a report, not a pass or fail.
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

from pattern_arrays import dol, walkforward  # noqa: E402
from pattern_arrays.data import load_bars, load_no_entry  # noqa: E402

START = "2018-01-01"
# The most recent 12 months are withheld from all fitting and evaluation.
END = "2025-10-17"
OUT = ROOT / "results"
ROWS_PQ = OUT / "arrayA_rows.parquet"


def build(bars_path, no_entry_path) -> tuple[pd.DataFrame, dict]:
    df1m = load_bars(bars_path, START, END)
    rows, counts = dol.build_rows(df1m, load_no_entry(no_entry_path))
    return rows, counts


def audit(rows: pd.DataFrame, counts: dict) -> pd.DataFrame:
    """Independent episodes = resolved (event/invalidated) rows; one per candidate at most,
    because a resolution ends the candidate's episodes. Coverage reported per year."""
    r = rows.copy()
    r["year"] = r["t_dec"].dt.year
    tab = r.pivot_table(index="year", columns="label", values="cand_id", aggfunc="count").fillna(0).astype(int)
    tab["rows"] = tab.sum(axis=1)
    tab["base_rate_ev_vs_inv"] = (tab.get("event", 0) / (tab.get("event", 0) + tab.get("invalidated", 0))).round(4)
    return tab


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars", required=True, help="Parquet/CSV of 1-minute bars")
    ap.add_argument("--no-entry", default=None, help="optional no-entry session calendar CSV")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if ROWS_PQ.exists() and not args.rebuild:
        rows = pd.read_parquet(ROWS_PQ)
        counts = json.loads((OUT / "arrayA_counts.json").read_text())
    else:
        rows, counts = build(args.bars, args.no_entry)
        rows.to_parquet(ROWS_PQ, index=False)
        (OUT / "arrayA_counts.json").write_text(json.dumps(counts, indent=2))
    print("counts:", counts)
    tab = audit(rows, counts)
    tab.to_csv(OUT / "arrayA_audit_by_year.csv")
    print(tab.to_string())

    res = walkforward.run(rows, START, END)
    res["oos"].to_parquet(OUT / "arrayA_oos_scores.parquet", index=False)
    pw = pd.DataFrame([{k: v for k, v in w.items() if k != "coef"} for w in res["per_window"]])
    pw.to_csv(OUT / "arrayA_per_window.csv", index=False)
    res["decile"].to_csv(OUT / "arrayA_decile.csv", index=False)
    res["decile_first_rows"].to_csv(OUT / "arrayA_decile_first_rows.csv", index=False)
    res["reliability"].to_csv(OUT / "arrayA_reliability.csv", index=False)
    summary = {
        "pooled": res["pooled"],
        "pooled_first_rows": res["pooled_first_rows"],
        "windows": len([w for w in res["per_window"] if not w.get("skipped")]),
        "windows_skipped": len([w for w in res["per_window"] if w.get("skipped")]),
        "coefs_last_window": next((w["coef"] for w in reversed(res["per_window"]) if not w.get("skipped")), None),
        "features": ["intercept"] + dol.FEATURES,
        "data": {"start": START, "end": END, "built": str(date.today())},
    }
    (OUT / "arrayA_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({"pooled": res["pooled"], "pooled_first_rows": res["pooled_first_rows"]}, indent=2))
    print(pw.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

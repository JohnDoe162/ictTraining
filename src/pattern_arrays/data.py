"""Bar loading.

The research behind this repository runs on a licensed Databento feed (CME `GLBX.MDP3`,
`MGC.v.0`). That data is not redistributable, so **no price data ships here**. Supply your own
bars through `load_bars`.

Required columns, one row per 1-minute bar, ascending by time:

| column     | type                        | meaning                                  |
|------------|-----------------------------|------------------------------------------|
| `ts_event` | tz-aware UTC timestamp      | the bar's OPEN time                      |
| `open`     | float                       |                                          |
| `high`     | float                       |                                          |
| `low`      | float                       |                                          |
| `close`    | float                       |                                          |
| `volume`   | int                         | not used by the model, kept for parity   |
| `contract` | str                         | e.g. `MGCZ5`; a change marks a roll      |

`contract` matters: the labeller refuses to carry a level across a contract change, so a
single-contract or back-adjusted series will not reproduce the published numbers exactly.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED = ["ts_event", "open", "high", "low", "close", "contract"]


def load_bars(path: str | Path, start: str, end: str) -> pd.DataFrame:
    """Load 1-minute bars in [start, end) from a Parquet or CSV file."""
    path = Path(path)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True)
    s, e = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    out = df[(df["ts_event"] >= s) & (df["ts_event"] < e)]
    return out.sort_values("ts_event").reset_index(drop=True)


def load_no_entry(path: str | Path | None) -> set:
    """Optional calendar of sessions that produce no decision rows: contract-roll crossover
    days, vendor-degraded days and holiday early closes. CSV with a `trading_day` column.
    Pass None to disable; the published numbers were produced WITH this calendar."""
    if path is None:
        return set()
    df = pd.read_csv(path)
    return set(pd.to_datetime(df["trading_day"]).dt.tz_localize("America/New_York").dt.normalize())

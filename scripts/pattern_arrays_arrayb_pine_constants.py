"""Write the fitted Array B constants into tradingview/sweep_model_mgc.pine.

Pine cannot read a file, so the weights are hardcoded. Generating them from the fitted JSON
removes the chance of a typo, and tests/test_pattern_arrays_pine.py fails if the two ever drift.

Run after scripts/pattern_arrays_arrayb_pine_parity.py:
    uv run python scripts/pattern_arrays_arrayb_pine_constants.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PINE = ROOT / "tradingview" / "sweep_model_mgc.pine"
MODEL = ROOT / "model" / "arrayB_model_pine.json"

PLAIN = {
    "sweep_depth_atr": "how far past the level the sweep ran, in ATR",
    "sweep_depth_ticks": "the same in ticks",
    "close_back_frac": "how far back inside the level the sweep bar closed (the rejection)",
    "sweep_bar_range_atr": "how violent the sweep bar was",
    "left_strength": "swing strength: earlier bars the pivot towers over",
    "prominence_atr": "swing strength: sharpness against its two neighbours",
    "age_bars": "age of the level, in 5m bars",
    "tests_held": "approaches that came close but did not break it",
    "atr_norm": "volatility regime",
    "rvol_sweep": "volume of the sweep bar against its recent median",
    "room_to_opposite_atr": "room to the opposite swing",
    "level_rank_session": "where the level sits in the session's range",
    "range_pos": "position of price in the recent range",
    "minutes_to_cap": "minutes left before the 17:00 cap",
    "side": "side: +1 swept high, -1 swept low",
    "bucket_london": "London session",
    "bucket_ny_am": "New York morning",
    "bucket_ny_pm": "New York afternoon",
    "x_atr": "GRID INPUT: how far back inside counts",
    "y_atr": "GRID INPUT: how much heat is allowed",
}


def main() -> int:
    m = json.loads(MODEL.read_text())
    lines = [f"C_INT = {m['intercept']:.6f}"]
    for i, f in enumerate(m["features"], start=1):
        lines.append(f"// {PLAIN.get(f, f)}")
        lines.append(f"W_{i:02d} = {m['coef'][i - 1]:>12.6f}")
        lines.append(f"M_{i:02d} = {m['mean'][i - 1]:>12.6f}")
        lines.append(f"S_{i:02d} = {m['std'][i - 1]:>12.6f}")
    block = "\n".join(lines)
    text = PINE.read_text(encoding="utf-8")
    if "// PLACEHOLDER_CONSTANTS" in text:
        text = text.replace("// PLACEHOLDER_CONSTANTS", block)
    else:
        text = re.sub(
            r"C_INT = -?\d+\.\d+\n(?:(?://[^\n]*\n)?[WMS]_\d{2} =\s*-?\d+\.\d+\n)+",
            block + "\n",
            text,
            count=1,
        )
    PINE.write_text(text, encoding="utf-8")
    print(f"wrote {len(m['features'])} feature blocks into {PINE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

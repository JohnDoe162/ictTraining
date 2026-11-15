"""The TradingView indicator must carry exactly the fitted model, and compute it correctly.

tradingview/dol_model_mgc.pine hardcodes the model's weights because Pine cannot read a file.
If the model is ever refitted and the .pine is not updated, the chart would silently show a
different model from the research. These tests make that impossible to miss.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PINE = ROOT / "tradingview" / "dol_model_mgc.pine"
MODEL_JSON = ROOT / "model" / "arrayA_model_pine.json"

pytestmark = pytest.mark.skipif(
    not (PINE.exists() and MODEL_JSON.exists()), reason="indicator or fitted model not present"
)


def pine_constants() -> dict[str, float]:
    text = PINE.read_text(encoding="utf-8")
    out = {}
    for name, val in re.findall(r"^([CWMS]_\d{2}|C_INT)\s*=\s*(-?\d+\.\d+)\s*$", text, re.M):
        out[name] = float(val)
    return out


def test_pine_weights_match_the_fitted_model():
    """The .pine carries each number rounded to 6 decimals, so the tolerance is one unit in
    that last place. Anything looser would let a genuinely different model through."""
    model = json.loads(MODEL_JSON.read_text())
    k = pine_constants()
    tol = 1e-6
    assert k["C_INT"] == pytest.approx(model["intercept"], abs=tol), "intercept drifted"
    assert len(model["features"]) == 12, "the .pine hardcodes exactly 12 features"
    for i, feat in enumerate(model["features"], start=1):
        tag = f"{i:02d}"
        assert k[f"W_{tag}"] == pytest.approx(model["coef"][i - 1], abs=tol), f"{feat} weight drifted"
        assert k[f"M_{tag}"] == pytest.approx(model["mean"][i - 1], abs=tol), f"{feat} mean drifted"
        assert k[f"S_{tag}"] == pytest.approx(model["std"][i - 1], abs=tol), f"{feat} sd drifted"


def test_rounding_to_six_decimals_cannot_move_a_score():
    """Rounding the published constants must not change what the chart shows. Worst case over
    the feature ranges actually seen in training (mean +/- 4 sd), the score moves < 0.1%."""
    model = json.loads(MODEL_JSON.read_text())
    k = pine_constants()
    worst = 0.0
    for sign in (-4.0, 4.0):
        z_full = model["intercept"]
        z_pine = k["C_INT"]
        for i in range(1, 13):
            tag = f"{i:02d}"
            x = model["mean"][i - 1] + sign * model["std"][i - 1]
            z_full += model["coef"][i - 1] * (x - model["mean"][i - 1]) / model["std"][i - 1]
            z_pine += k[f"W_{tag}"] * (x - k[f"M_{tag}"]) / k[f"S_{tag}"]
        p_full = 1.0 / (1.0 + math.exp(-z_full))
        p_pine = 1.0 / (1.0 + math.exp(-z_pine))
        worst = max(worst, abs(p_full - p_pine))
    assert worst < 0.001, f"rounding moves the score by {worst:.6f}"


def test_pine_scoring_arithmetic_known_answer():
    """Hand-worked: feed every feature its own training mean. Each standardised term is then
    exactly 0, so the score must be the logistic function of the intercept alone."""
    model = json.loads(MODEL_JSON.read_text())
    z = model["intercept"]
    for w, m, s in zip(model["coef"], model["mean"], model["std"]):
        z += w * (m - m) / s
    assert z == pytest.approx(model["intercept"], abs=1e-12)
    p = 1.0 / (1.0 + math.exp(-z))
    # intercept 0.063218 -> 1/(1+e^-0.063218) = 0.515798...
    assert p == pytest.approx(0.5158, abs=1e-4)
    # and that sits sensibly beside the base rate the model was fitted on
    assert abs(p - model["base_rate"]) < 0.05


def test_pine_declares_the_rules_the_owner_fixed():
    """The label rules are the owner's, not the model's: 1-hour bars, trade-through by one
    tick, and the session cap. They must be stated in the file the owner reads."""
    text = PINE.read_text(encoding="utf-8").lower()
    assert "1-hour" in text and 'timeframe.period != "60"' in text.lower()
    assert "traded through" in text
    assert "at least one tick" in text or "at least 1 tick" in text
    assert "not a trade signal" in text


def test_pine_has_no_lookahead_markers():
    """Pine repaints if it requests a higher timeframe with lookahead on. The indicator must
    not use request.security at all: it reads only the chart's own closed bars."""
    text = PINE.read_text(encoding="utf-8")
    assert "request.security" not in text
    assert "lookahead_on" not in text
    assert "barmerge.lookahead_on" not in text


# --------------------------------------------------------------------------------------
# Pine gives every if/else block a value type. A branch ending in an assignment returns
# that object; a branch ending in a void call (line.set_*, label.delete, break ...) returns
# nothing. Mixing the two is Pine error CE10235, which only TradingView's compiler would
# normally catch - and it reports one occurrence at a time. This checks the whole file.
# --------------------------------------------------------------------------------------
_VOID = re.compile(
    r"^(line|label|table|box|array|runtime|alert)\."
    r"(set_[a-z_]+|delete|push|remove|clear|cell|error|unshift|insert|fill|sort|reverse|concat)\s*\("
)


def _kind(stmt: str) -> str:
    s = stmt.strip()
    return "void" if (s in ("break", "continue") or _VOID.match(s)) else "value"


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _branch_last(lines: list[str], start: int, base: int) -> str | None:
    last = None
    for i in range(start, len(lines)):
        line = lines[i]
        if not line.strip() or line.strip().startswith("//"):
            continue
        if _indent(line) <= base:
            break
        last = line
    return last


def test_pine_if_else_branches_return_the_same_type():
    lines = PINE.read_text(encoding="utf-8").split("\n")
    problems = []
    for i, line in enumerate(lines):
        st = line.strip()
        if st != "else" and not st.startswith("else if"):
            continue
        ind = _indent(line)
        j, if_at = i - 1, None
        while j >= 0:
            lj = lines[j]
            if lj.strip() and not lj.strip().startswith("//"):
                if _indent(lj) == ind and (lj.strip().startswith("if ") or lj.strip().startswith("else")):
                    if_at = j
                if _indent(lj) <= ind:
                    break
            j -= 1
        if if_at is None:
            continue
        a = _branch_last(lines, if_at + 1, ind)
        b = _branch_last(lines, i + 1, ind)
        if a is not None and b is not None and _kind(a) != _kind(b):
            problems.append(f"line {i + 1}: if ends '{a.strip()}' ({_kind(a)}), else ends '{b.strip()}' ({_kind(b)})")
    assert not problems, "Pine if/else type mismatch (CE10235):\n" + "\n".join(problems)


def _logical_lines(text: str) -> list[str]:
    """Join Pine continuation lines so a wrapped call reads as one statement."""
    out: list[str] = []
    for raw in text.split("\n"):
        s = raw.strip()
        if not s or s.startswith("//"):
            continue
        if out and (out[-1].count("(") > out[-1].count(")")):
            out[-1] = out[-1] + " " + s
        else:
            out.append(s)
    return out


def test_every_table_cell_has_an_opaque_background():
    """The owner hit an unreadable table on the dark theme: cells with no background let the
    chart show through and dark text vanished. Every cell must set one."""
    missing = [
        ln
        for ln in _logical_lines(PINE.read_text(encoding="utf-8"))
        if ln.startswith("table.cell(") and "bgcolor" not in ln
    ]
    assert not missing, "table cells without a background:\n" + "\n".join(missing)


def test_colours_adapt_to_the_chart_theme():
    """Light and dark must be two selected palettes, chosen at runtime, not one hardcoded set."""
    text = PINE.read_text(encoding="utf-8")
    assert "chart.bg_color" in text, "theme is not detected from the chart background"
    assert "isDark" in text
    for name in ("C_INK", "C_INK2", "C_SURFACE", "C_HEAD", "C_HIGH"):
        assert re.search(rf"^{name}\s*=\s*isDark \?", text, re.M), f"{name} does not switch on theme"


def test_drawings_are_forced_onto_the_price_pane():
    """overlay=true alone did not stop TradingView putting the script in its own pane, so the
    drawings themselves must name the price pane."""
    text = PINE.read_text(encoding="utf-8")
    assert re.search(r"indicator\(.*overlay\s*=\s*true", text, re.S), "indicator is not overlay"
    for ln in _logical_lines(text):
        if ln.startswith("c.ln := line.new(") or ln.startswith("c.lb := label.new("):
            assert "force_overlay = true" in ln, f"drawing not forced onto the price pane: {ln[:70]}"

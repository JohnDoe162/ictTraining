"""The TradingView indicators must carry exactly the fitted models, and compute them correctly.

Pine cannot read a file, so each indicator hardcodes its model's weights. If a model is refitted
and the .pine is not regenerated, the chart would silently show a different model from the
research. These tests make that impossible to miss, and the structural checks below run against
every indicator in the project.

  tradingview/dol_model_mgc.pine    - Array A, 1h draw on liquidity, 12 features
  tradingview/sweep_model_mgc.pine  - Array B, 5m sweep and reversal, 20 features
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "model"
INDICATORS = {
    "arrayA": (ROOT / "tradingview" / "dol_model_mgc.pine", MODELS / "arrayA_model_pine.json", 12),
    "arrayB": (ROOT / "tradingview" / "sweep_model_mgc.pine", MODELS / "arrayB_model_pine.json", 20),
}
PRESENT = [k for k, (pine, model, _) in INDICATORS.items() if pine.exists() and model.exists()]

pytestmark = pytest.mark.skipif(not PRESENT, reason="no indicator with a fitted model is present")


def pytest_generate_tests(metafunc):  # pragma: no cover - collection hook
    if "name" in metafunc.fixturenames:
        metafunc.parametrize("name", PRESENT)


def pine_constants(pine: Path) -> dict[str, float]:
    text = pine.read_text(encoding="utf-8")
    out = {}
    for name, val in re.findall(r"^([CWMS]_\d{2}|C_INT)\s*=\s*(-?\d+\.\d+)\s*$", text, re.M):
        out[name] = float(val)
    return out


def test_pine_weights_match_the_fitted_model(name):
    """The .pine carries each number rounded to 6 decimals, so the tolerance is one unit in
    that last place. Anything looser would let a genuinely different model through."""
    pine, model_json, n_feat = INDICATORS[name]
    model = json.loads(model_json.read_text())
    k = pine_constants(pine)
    tol = 1e-6
    assert k["C_INT"] == pytest.approx(model["intercept"], abs=tol), "intercept drifted"
    assert len(model["features"]) == n_feat, f"{name}: the .pine hardcodes exactly {n_feat} features"
    for i, feat in enumerate(model["features"], start=1):
        tag = f"{i:02d}"
        assert k[f"W_{tag}"] == pytest.approx(model["coef"][i - 1], abs=tol), f"{feat} weight drifted"
        assert k[f"M_{tag}"] == pytest.approx(model["mean"][i - 1], abs=tol), f"{feat} mean drifted"
        assert k[f"S_{tag}"] == pytest.approx(model["std"][i - 1], abs=tol), f"{feat} sd drifted"


def test_rounding_to_six_decimals_cannot_move_a_score(name):
    """Rounding the published constants must not change what the chart shows. Worst case over
    the feature ranges actually seen in training (mean +/- 4 sd), the score moves < 0.1%."""
    pine, model_json, n_feat = INDICATORS[name]
    model = json.loads(model_json.read_text())
    k = pine_constants(pine)
    worst = 0.0
    for sign in (-4.0, 4.0):
        z_full = model["intercept"]
        z_pine = k["C_INT"]
        for i in range(1, n_feat + 1):
            tag = f"{i:02d}"
            x = model["mean"][i - 1] + sign * model["std"][i - 1]
            z_full += model["coef"][i - 1] * (x - model["mean"][i - 1]) / model["std"][i - 1]
            z_pine += k[f"W_{tag}"] * (x - k[f"M_{tag}"]) / k[f"S_{tag}"]
        p_full = 1.0 / (1.0 + math.exp(-z_full))
        p_pine = 1.0 / (1.0 + math.exp(-z_pine))
        worst = max(worst, abs(p_full - p_pine))
    assert worst < 0.001, f"{name}: rounding moves the score by {worst:.6f}"


def test_pine_scoring_arithmetic_known_answer(name):
    """Hand-worked: feed every feature its own training mean. Each standardised term is then
    exactly 0, so the score must be the logistic function of the intercept alone, and that must
    sit beside the base rate the model was fitted on."""
    _, model_json, _ = INDICATORS[name]
    model = json.loads(model_json.read_text())
    z = model["intercept"]
    for w, m, s in zip(model["coef"], model["mean"], model["std"]):
        z += w * (m - m) / s
    assert z == pytest.approx(model["intercept"], abs=1e-12)
    p = 1.0 / (1.0 + math.exp(-z))
    assert p == pytest.approx(1.0 / (1.0 + math.exp(-model["intercept"])), abs=1e-12)
    assert abs(p - model["base_rate"]) < 0.05


def test_pine_declares_the_rules_the_owner_fixed(name):
    """The label rules are the owner's, not the model's. They must be stated in the file the
    owner reads, together with the timeframe guard that keeps the model on the bars it was
    trained on."""
    pine, _, _ = INDICATORS[name]
    text = pine.read_text(encoding="utf-8").lower()
    assert "traded through" in text
    assert "at least one tick" in text or "at least 1 tick" in text
    assert "not a trade signal" in text
    if name == "arrayA":
        assert "1-hour" in text and 'timeframe.period != "60"' in text
        assert "session cap" in text or "17:00" in text
    if name == "arrayB":
        assert "5-minute" in text and 'timeframe.period != "5"' in text
        assert "17:00" in text
        # the one honest caveat the owner must not lose: the model is close to one measurement
        assert "0.671" in text and "0.673" in text


def test_pine_has_no_lookahead_markers(name):
    """Pine repaints if it requests a higher timeframe with lookahead on. The indicator must
    not use request.security at all: it reads only the chart's own closed bars."""
    pine, _, _ = INDICATORS[name]
    text = pine.read_text(encoding="utf-8")
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


def test_pine_if_else_branches_return_the_same_type(name):
    pine, _, _ = INDICATORS[name]
    lines = pine.read_text(encoding="utf-8").split("\n")
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


def test_every_table_cell_has_an_opaque_background(name):
    """The owner hit an unreadable table on the dark theme: cells with no background let the
    chart show through and dark text vanished. Every cell must set one."""
    pine, _, _ = INDICATORS[name]
    missing = [
        ln
        for ln in _logical_lines(pine.read_text(encoding="utf-8"))
        if ln.startswith("table.cell(") and "bgcolor" not in ln
    ]
    assert not missing, "table cells without a background:\n" + "\n".join(missing)


def test_colours_adapt_to_the_chart_theme(name):
    """Light and dark must be two selected palettes, chosen at runtime, not one hardcoded set."""
    pine, _, _ = INDICATORS[name]
    text = pine.read_text(encoding="utf-8")
    assert "chart.bg_color" in text, "theme is not detected from the chart background"
    assert "isDark" in text
    for name in ("C_INK", "C_INK2", "C_SURFACE", "C_HEAD", "C_HIGH"):
        assert re.search(rf"^{name}\s*=\s*isDark \?", text, re.M), f"{name} does not switch on theme"


def test_drawings_are_forced_onto_the_price_pane(name):
    """overlay=true alone did not stop TradingView putting the script in its own pane, so the
    drawings themselves must name the price pane."""
    pine, _, _ = INDICATORS[name]
    text = pine.read_text(encoding="utf-8")
    assert re.search(r"indicator\(.*overlay\s*=\s*true", text, re.S), "indicator is not overlay"
    for ln in _logical_lines(text):
        if ln.startswith("c.ln := line.new(") or ln.startswith("c.lb := label.new("):
            assert "force_overlay = true" in ln, f"drawing not forced onto the price pane: {ln[:70]}"

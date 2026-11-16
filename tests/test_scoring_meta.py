"""math-verify version in meta.json and the report's "scoring" summary card.

Loop 2026-11-16-006.
"""
import html as html_lib
import importlib.metadata
import json
import re
import sys

import pytest

import skillopt.runlog
from helpers import element_by_id, find_elements, read_json
from skillopt.benchmarks.math_answer import MATH_VERIFY_DIST, math_verify_version
from skillopt.report import load_run, render_report
from skillopt.runlog import RunDir
from test_report import SynRun


# --- helpers ---------------------------------------------------------------------------

def _text(el) -> str:
    return " ".join(html_lib.unescape(re.sub(r"<[^>]*>", " ", el.html)).split())


def _cards(html):
    return [c for c in find_elements(element_by_id(html, "summary").html, cls="card", tag="div")
            if "data-card" in c.attrs]


def _scoring_card(html):
    found = [c for c in _cards(html) if c.attrs.get("data-card") == "scoring"]
    assert len(found) == 1, len(found)
    return found[0]


def _part(card, cls):
    found = find_elements(card.html, cls=cls, tag="div")
    assert len(found) == 1, (cls, len(found))
    return _text(found[0])


def _syn(tmp_path, meta=None, meta_raw=None):
    run = SynRun(tmp_path)
    run.skill(0, "a")
    if meta is not None:
        run.raw("meta.json", json.dumps(meta))
    elif meta_raw is not None:
        run.raw("meta.json", meta_raw)
    return run


def _card_value(run) -> str:
    html = render_report(load_run(run.path))
    return _part(_scoring_card(html), "card-value")


# --- AC1-3: math_verify_version -----------------------------------------------------------

def test_dist_constant():
    assert MATH_VERIFY_DIST == "math-verify"


def test_version_returned(monkeypatch):
    asked = []

    def fake(name):
        asked.append(name)
        return "0.9.0"

    monkeypatch.setattr(importlib.metadata, "version", fake)
    assert math_verify_version() == "0.9.0"
    assert asked and all(a == "math-verify" for a in asked)


def test_version_not_cached(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "1.0.0")
    assert math_verify_version() == "1.0.0"
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "2.0.0")
    assert math_verify_version() == "2.0.0"


def test_version_package_not_found(monkeypatch):
    def fake(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", fake)
    assert math_verify_version() is None


@pytest.mark.parametrize("exc", [RuntimeError("boom"), ValueError("bad"), OSError("io")],
                         ids=["runtime", "value", "os"])
def test_version_other_exception(monkeypatch, exc):
    def fake(name):
        raise exc

    monkeypatch.setattr(importlib.metadata, "version", fake)
    assert math_verify_version() is None


def test_version_does_not_import_math_verify_or_sympy(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.9.0")
    before = set(sys.modules)
    math_verify_version()
    added = set(sys.modules) - before
    assert "math_verify" not in added
    assert "sympy" not in added
    assert not [m for m in added if m.startswith("math_verify.") or m.startswith("sympy.")], added


# --- AC4: RunDir meta.json --------------------------------------------------------------

def test_rundir_meta_scoring_version(tmp_path, monkeypatch):
    monkeypatch.setattr(skillopt.runlog, "math_verify_version", lambda: "0.9.0")
    rd = RunDir(tmp_path, "x", {"seed": 7})
    meta = read_json(rd.path / "meta.json")
    assert meta["scoring"] == {"math_verify": "0.9.0"}
    for key in ("seed", "created_at", "argv", "python"):
        assert key in meta, key
    assert meta["seed"] == 7


def test_rundir_meta_scoring_none(tmp_path, monkeypatch):
    monkeypatch.setattr(skillopt.runlog, "math_verify_version", lambda: None)
    rd = RunDir(tmp_path, "x", {"seed": 0})
    meta = read_json(rd.path / "meta.json")
    assert meta["scoring"] == {"math_verify": None}
    assert "scoring" in meta and meta["scoring"]["math_verify"] is None
    for key in ("seed", "created_at", "argv", "python"):
        assert key in meta, key


# --- AC6: load_run["scoring"] -------------------------------------------------------------

def test_load_run_scoring_dict(tmp_path):
    run = _syn(tmp_path, meta={"seed": 0, "scoring": {"math_verify": "0.9.0"}})
    assert load_run(run.path)["scoring"] == {"math_verify": "0.9.0"}


def test_load_run_scoring_null_version(tmp_path):
    run = _syn(tmp_path, meta={"scoring": {"math_verify": None}})
    assert load_run(run.path)["scoring"] == {"math_verify": None}


def test_load_run_scoring_no_meta(tmp_path):
    run = _syn(tmp_path)
    data = load_run(run.path)
    assert "scoring" in data
    assert data["scoring"] is None


def test_load_run_scoring_invalid_json(tmp_path):
    run = _syn(tmp_path, meta_raw="{not json")
    data = load_run(run.path)
    assert "scoring" in data
    assert data["scoring"] is None


def test_load_run_scoring_missing_key(tmp_path):
    run = _syn(tmp_path, meta={"seed": 0, "python": "3.14.0"})
    data = load_run(run.path)
    assert "scoring" in data
    assert data["scoring"] is None


@pytest.mark.parametrize("bad", ["0.9.0", ["math_verify"], 5, None], ids=["str", "list", "int", "null"])
def test_load_run_scoring_non_dict(tmp_path, bad):
    run = _syn(tmp_path, meta={"scoring": bad})
    data = load_run(run.path)
    assert "scoring" in data
    assert data["scoring"] is None


# --- AC7-9: scoring card ------------------------------------------------------------------

def test_scoring_card_is_last(tmp_path):
    run = _syn(tmp_path, meta={"scoring": {"math_verify": "0.9.0"}})
    html = render_report(load_run(run.path))
    order = [c.attrs.get("data-card") for c in _cards(html)]
    assert order == ["round", "status", "skills", "target-usage", "optimizer-usage", "scoring"]
    assert order[-1] == "scoring"


def test_scoring_card_label(tmp_path):
    run = _syn(tmp_path, meta={"scoring": {"math_verify": "0.9.0"}})
    html = render_report(load_run(run.path))
    assert _part(_scoring_card(html), "card-label") == "Scoring"


def test_scoring_card_version(tmp_path):
    run = _syn(tmp_path, meta={"scoring": {"math_verify": "0.9.0"}})
    assert _card_value(run) == "math-verify: 0.9.0"


def test_scoring_card_not_installed(tmp_path):
    run = _syn(tmp_path, meta={"scoring": {"math_verify": None}})
    assert _card_value(run) == "math-verify: not installed"


def test_scoring_card_unknown_no_meta(tmp_path):
    assert _card_value(_syn(tmp_path)) == "math-verify: unknown"


def test_scoring_card_unknown_old_meta(tmp_path):
    run = _syn(tmp_path, meta={"seed": 0, "created_at": "2026-01-01T00:00:00", "argv": [], "python": "3.14.0"})
    assert _card_value(run) == "math-verify: unknown"


def test_scoring_card_unknown_scoring_without_math_verify(tmp_path):
    run = _syn(tmp_path, meta={"scoring": {}})
    assert _card_value(run) == "math-verify: unknown"


def test_scoring_card_unknown_invalid_json(tmp_path):
    run = _syn(tmp_path, meta_raw="{not json")
    assert _card_value(run) == "math-verify: unknown"


@pytest.mark.parametrize("bad", ["0.9.0", ["x"], 5], ids=["str", "list", "int"])
def test_scoring_card_unknown_non_dict_scoring(tmp_path, bad):
    run = _syn(tmp_path, meta={"scoring": bad})
    assert _card_value(run) == "math-verify: unknown"


@pytest.mark.parametrize("bad", [5, "", 1.5, ["0.9.0"], {"v": 1}, True],
                         ids=["int", "empty", "float", "list", "dict", "bool"])
def test_scoring_card_unknown_bad_version_value(tmp_path, bad):
    run = _syn(tmp_path, meta={"scoring": {"math_verify": bad}})
    assert _card_value(run) == "math-verify: unknown"


def test_scoring_card_unknown_from_data_dict(tmp_path):
    """render_report on data whose scoring is None (as load_run gives for old runs)."""
    run = _syn(tmp_path)
    data = load_run(run.path)
    data["scoring"] = None
    assert _part(_scoring_card(render_report(data)), "card-value") == "math-verify: unknown"


def test_scoring_card_escapes_version(tmp_path):
    run = _syn(tmp_path, meta={"scoring": {"math_verify": "<b>1</b>"}})
    html = render_report(load_run(run.path))
    card = _scoring_card(html)
    assert "<b>" not in card.html
    assert not find_elements(card.html, tag="b")
    assert _part(card, "card-value") == "math-verify: <b>1</b>"

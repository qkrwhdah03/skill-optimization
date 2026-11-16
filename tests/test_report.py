"""Run report: skillopt.report load_run / render_report / write_report / main on synthetic run dirs.

Loops 2026-11-16-001 (data contracts) and 2026-11-16-002 (tree-first interactive layout).
"""
import html as html_lib
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from helpers import PROJECT_ROOT, element_by_id, find_elements, html_elements
from skillopt.report import (
    MAX_FAILURES_SHOWN,
    MAX_TEXT_CHARS,
    REFRESH_SECONDS,
    REPORT_FILENAME,
    load_run,
    main,
    render_report,
    write_report,
)

REFRESH_META = '<meta http-equiv="refresh" content="5">'


# --- synthetic run builder --------------------------------------------------------

def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _bench_summary(sid, bench, n, acc, tok):
    return {"skill_id": sid, "benchmark": bench, "split": "train", "n": n, "accuracy": acc,
            "avg_input_tokens": 50.0, "avg_output_tokens": tok, "avg_reasoning_tokens": 0.0,
            "avg_total_tokens": 50.0 + tok, "avg_latency": 0.1, "num_truncated": 0, "truncation_rate": 0.0}


def _item(sid, bench, i, correct, question=None):
    return {"example_id": f"ex{i}", "prediction": "1" if correct else "2", "gold": "1", "correct": correct,
            "usage": {"input_tokens": 50, "output_tokens": 10, "reasoning_tokens": 0, "total_tokens": 60},
            "latency": 0.1, "response_text": "The answer is 1." if correct else "The answer is 2.",
            "reasoning_text": "", "question": question or f"q-{sid}-{bench}-{i}", "truncated": False}


class SynRun:
    """Writes a run directory file by file in the saved format (no skillopt code involved)."""

    def __init__(self, root: Path, name: str = "20261116-000000_syn", optimizer: str = "opro",
                 max_iterations: int = 3) -> None:
        self.path = root / name
        self.path.mkdir(parents=True)
        with open(self.path / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump({"seed": 0, "optimizer": {"name": optimizer}, "max_iterations": max_iterations}, f)
        self.summaries: dict[int, list] = {}

    def rdir(self, rnd: int) -> Path:
        return self.path / "rounds" / f"{rnd:03d}"

    def skill(self, rnd, sid, benches=None, instruction=None, origin=None, parent_id=None, num_tokens=None,
              fails=None, few_shot=None, metadata=None, skill_json=True, jsonl=True, items=None):
        """``benches``: {bench: (acc, avg_output_tokens)}; ``fails``: {bench: number of incorrect items}."""
        benches = benches if benches is not None else {"gsm8k": (0.5, 10.0)}
        fails = fails or {}
        d = self.rdir(rnd)
        entry = {"skill_id": sid, "split": "train", "benchmarks": {}}
        for bench, (acc, tok) in benches.items():
            k = fails.get(bench, 0)
            n = max(4, k)
            entry["benchmarks"][bench] = _bench_summary(sid, bench, n, acc, tok)
            if jsonl:
                rows = (items or {}).get(bench)
                if rows is None:
                    rows = [_item(sid, bench, i, i < n - k) for i in range(n)]
                p = d / "results" / bench / f"{sid}.jsonl"
                p.parent.mkdir(parents=True, exist_ok=True)
                with open(p, "w", encoding="utf-8") as f:
                    for row in rows:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if skill_json:
            _write_json(d / "skills" / f"{sid}.json", {
                "id": sid,
                "instruction": instruction if instruction is not None else f"Instruction for {sid}.",
                "few_shot": list(few_shot or []),
                "parent_id": parent_id,
                "metadata": dict(metadata or {}),
                "origin": origin,
                "num_tokens": num_tokens,
            })
        self.summaries.setdefault(rnd, []).append(entry)
        _write_json(d / "summary.json", self.summaries[rnd])
        return self

    def round_usage(self, rnd, obj):
        _write_json(self.rdir(rnd) / "usage.json", obj)
        return self

    def usage(self, role, obj):
        _write_json(self.path / "usage" / f"{role}.json", obj)
        return self

    def finish(self, stop_reason="max_iterations", rounds=None, max_iterations=None):
        _write_json(self.path / "run.json", {
            "rounds": rounds if rounds is not None else len(self.summaries),
            "stop_reason": stop_reason,
            "max_iterations": max_iterations if max_iterations is not None else 3,
        })
        return self

    def raw(self, rel, text):
        p = self.path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        return self


def _origin(parents, operation="mutate", note=""):
    return {"parents": list(parents), "operation": operation, "note": note}


def _usage(total=913):
    return {"calls": 3, "input_tokens": 100, "output_tokens": total - 100, "reasoning_tokens": 0,
            "total_tokens": total, "latency": 1.5}


def _skills(data, rnd_index):
    for r in data["rounds"]:
        if r["index"] == rnd_index:
            return {s["id"]: s for s in r["skills"]}
    raise KeyError(rnd_index)


def _lineage_run(root):
    """Re-evaluated ids, multi-parent, parent_id fallback, unresolved and same-round parents."""
    run = SynRun(root)
    run.skill(0, "a", instruction="Think step by step.", origin=_origin([], "initial"))
    run.skill(0, "x", instruction="Answer directly.", origin=_origin([], "initial"))
    run.skill(1, "a", instruction="Think step by step.", origin=_origin([], "reeval"))
    run.skill(1, "b", instruction="Think briefly step by step.", origin=_origin(["a"], "rewrite"))
    run.skill(1, "b2", instruction="Answer directly now.", origin=_origin(["x"], "rewrite"))
    run.skill(2, "c", instruction="Think step by step quickly.", origin=_origin(["a", "x"], "crossover"))
    run.skill(2, "d", instruction="Think briefly.", parent_id="b")
    run.skill(2, "e", instruction="Fresh idea.")
    run.skill(2, "g", instruction="Orphan text.", origin=_origin(["ghost"], "mutate"))
    run.skill(2, "h", instruction="Same round parent.", origin=_origin(["c"], "mutate"))
    run.skill(2, "k", instruction="Answer directly please.", origin=_origin(["x"], "mutate"))
    return run

# --- AC1: constants ------------------------------------------------------------------

def test_constants():
    assert REPORT_FILENAME == "report.html"
    assert REFRESH_SECONDS == 5
    assert MAX_FAILURES_SHOWN == 3
    assert MAX_TEXT_CHARS == 300


# --- AC2: load_run basics -------------------------------------------------------

def test_load_run_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_run(tmp_path / "nope")


def test_load_run_file_path(tmp_path):
    f = tmp_path / "file.txt"
    f.write_text("x", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        load_run(f)


def test_load_run_no_rounds(tmp_path):
    run = SynRun(tmp_path)
    data = load_run(run.path)
    assert data["rounds"] == []
    assert data["benchmarks"] == []


def test_load_run_name_and_absolute_dir(tmp_path, monkeypatch):
    run = SynRun(tmp_path, name="20261116-120000_abc")
    run.skill(0, "a")
    monkeypatch.chdir(tmp_path)
    data = load_run("20261116-120000_abc")
    assert data["name"] == "20261116-120000_abc"
    assert isinstance(data["run_dir"], str)
    assert Path(data["run_dir"]).is_absolute()
    assert Path(data["run_dir"]).resolve() == run.path.resolve()


def test_load_run_running(tmp_path):
    run = SynRun(tmp_path, optimizer="my_opt", max_iterations=7)
    run.skill(0, "a")
    data = load_run(run.path)
    assert data["optimizer"] == "my_opt"
    assert data["finished"] is False
    assert data["stop_reason"] is None
    assert data["max_iterations"] == 7


def test_load_run_finished(tmp_path):
    run = SynRun(tmp_path, max_iterations=7)
    run.skill(0, "a")
    run.finish(stop_reason="empty_update", rounds=1, max_iterations=2)
    data = load_run(run.path)
    assert data["finished"] is True
    assert data["stop_reason"] == "empty_update"
    assert data["max_iterations"] == 2


def test_load_run_accepts_str(tmp_path):
    run = SynRun(tmp_path)
    run.skill(0, "a")
    assert [r["index"] for r in load_run(str(run.path))["rounds"]] == [0]


# --- AC2: rounds, order, benchmarks ---------------------------------------------

def test_rounds_numeric_order_and_skips(tmp_path):
    run = SynRun(tmp_path)
    run.skill(0, "a").skill(1, "b").skill(10, "c")
    (run.path / "rounds" / "tmp").mkdir()
    _write_json(run.path / "rounds" / "tmp" / "summary.json", [])
    (run.path / "rounds" / "002" / "skills").mkdir(parents=True)  # no summary.json
    run.raw("rounds/003/summary.json", "{not json")  # broken summary.json
    data = load_run(run.path)
    assert [r["index"] for r in data["rounds"]] == [0, 1, 10]


def test_round_zero_only(tmp_path):
    run = SynRun(tmp_path)
    run.skill(0, "a")
    data = load_run(run.path)
    assert len(data["rounds"]) == 1
    assert [s["id"] for s in data["rounds"][0]["skills"]] == ["a"]


def test_skill_order_is_summary_order(tmp_path):
    run = SynRun(tmp_path)
    for sid in ("zeta", "alpha", "mid", "beta"):
        run.skill(0, sid)
    data = load_run(run.path)
    assert [s["id"] for s in data["rounds"][0]["skills"]] == ["zeta", "alpha", "mid", "beta"]


def test_benchmarks_first_seen_order(tmp_path):
    run = SynRun(tmp_path)
    run.skill(0, "a", benches={"zbench": (0.5, 1.0)})
    run.skill(0, "b", benches={"gsm8k": (0.5, 1.0), "zbench": (0.5, 1.0)})
    run.skill(1, "c", benches={"bbh": (0.5, 1.0), "gsm8k": (0.5, 1.0)})
    assert load_run(run.path)["benchmarks"] == ["zbench", "gsm8k", "bbh"]


# --- AC2: skill fields -------------------------------------------------------------

def test_skill_json_missing_defaults(tmp_path):
    run = SynRun(tmp_path)
    run.skill(0, "a", skill_json=False)
    s = _skills(load_run(run.path), 0)["a"]
    assert s["instruction"] == ""
    assert s["few_shot"] == []
    assert s["num_tokens"] is None
    assert s["parents"] == []
    assert s["operation"] is None
    assert s["note"] == ""


def test_skill_json_fields(tmp_path):
    run = SynRun(tmp_path)
    run.skill(0, "a", instruction="Solve it.", num_tokens=42,
              few_shot=[{"question": "1+1?", "answer": "2"}], metadata={"k": 1},
              origin=_origin([], "initial", "first draft"))
    s = _skills(load_run(run.path), 0)["a"]
    assert s["instruction"] == "Solve it."
    assert s["num_tokens"] == 42
    assert s["few_shot"] == [{"question": "1+1?", "answer": "2"}]
    assert s["metadata"] == {"k": 1}
    assert s["operation"] == "initial"
    assert s["note"] == "first draft"


def test_skill_without_origin_has_no_operation(tmp_path):
    run = SynRun(tmp_path)
    run.skill(0, "a", num_tokens=None)
    s = _skills(load_run(run.path), 0)["a"]
    assert s["operation"] is None
    assert s["note"] == ""
    assert s["num_tokens"] is None


# --- AC2: parents / children ------------------------------------------------------

def test_parents_resolution(tmp_path):
    data = load_run(_lineage_run(tmp_path).path)
    r0, r1, r2 = _skills(data, 0), _skills(data, 1), _skills(data, 2)
    assert r0["a"]["parents"] == []
    assert r1["a"]["parents"] == []
    assert r1["b"]["parents"] == [{"id": "a", "round": 0}]
    # most recent earlier round wins (a is in rounds 0 and 1); order of origin.parents kept
    assert r2["c"]["parents"] == [{"id": "a", "round": 1}, {"id": "x", "round": 0}]
    # no origin -> [parent_id]
    assert r2["d"]["parents"] == [{"id": "b", "round": 1}]
    # neither -> []
    assert r2["e"]["parents"] == []
    # outside the run -> round None
    assert r2["g"]["parents"] == [{"id": "ghost", "round": None}]
    # only earlier rounds count
    assert r2["h"]["parents"] == [{"id": "c", "round": None}]


def test_origin_parents_take_precedence_over_parent_id(tmp_path):
    run = SynRun(tmp_path)
    run.skill(0, "a").skill(0, "b")
    run.skill(1, "c", origin=_origin(["b", "a"], "crossover"), parent_id="a")
    assert _skills(load_run(run.path), 1)["c"]["parents"] == [{"id": "b", "round": 0}, {"id": "a", "round": 0}]


def test_children(tmp_path):
    data = load_run(_lineage_run(tmp_path).path)
    r0, r1, r2 = _skills(data, 0), _skills(data, 1), _skills(data, 2)
    # a@0 only gets b (c resolves a to round 1)
    assert r0["a"]["children"] == [{"id": "b", "round": 1}]
    assert r1["a"]["children"] == [{"id": "c", "round": 2}]
    # round order, then evaluation order
    assert r0["x"]["children"] == [{"id": "b2", "round": 1}, {"id": "c", "round": 2}, {"id": "k", "round": 2}]
    assert r1["b"]["children"] == [{"id": "d", "round": 2}]
    assert r1["b2"]["children"] == []
    # h's parent c is unresolved (same round) -> no child
    assert r2["c"]["children"] == []
    for sid in ("d", "e", "g", "h", "k"):
        assert r2[sid]["children"] == []


# --- AC2: results / failures (failures capped at MAX_FAILURES_SHOWN) ---------------------------------------------------------

def test_results_as_saved(tmp_path):
    run = SynRun(tmp_path)
    run.skill(0, "a", benches={"gsm8k": (0.25, 12.5), "bbh": (0.75, 3.0)})
    s = _skills(load_run(run.path), 0)["a"]
    assert s["results"] == run.summaries[0][0]["benchmarks"]


@pytest.mark.parametrize("k, shown", [(0, 0), (1, 1), (3, 3), (4, 3), (25, 3)])
def test_failures_cap(tmp_path, k, shown):
    run = SynRun(tmp_path)
    run.skill(0, "a", fails={"gsm8k": k})
    s = _skills(load_run(run.path), 0)["a"]
    assert len(s["failures"]["gsm8k"]) == shown
    assert s["num_failures"]["gsm8k"] == k
    assert all(f["correct"] is False for f in s["failures"]["gsm8k"])


def test_failures_saved_order(tmp_path):
    rows = []
    for i in range(30):
        rows.append(_item("a", "gsm8k", i, correct=(i % 3 == 0)))  # 20 failures, interleaved
    run = SynRun(tmp_path)
    run.skill(0, "a", items={"gsm8k": rows})
    s = _skills(load_run(run.path), 0)["a"]
    expected = [r["question"] for r in rows if not r["correct"]][:MAX_FAILURES_SHOWN]
    assert [f["question"] for f in s["failures"]["gsm8k"]] == expected
    assert s["num_failures"]["gsm8k"] == 20


def test_failures_jsonl_missing(tmp_path):
    run = SynRun(tmp_path)
    run.skill(0, "a", jsonl=False)
    s = _skills(load_run(run.path), 0)["a"]
    assert s["failures"]["gsm8k"] == []
    assert s["num_failures"]["gsm8k"] == 0


# --- AC2: usage ----------------------------------------------------------------------

def test_usage(tmp_path):
    run = SynRun(tmp_path)
    run.skill(0, "a").skill(1, "b")
    run.usage("target", _usage(913))
    run.round_usage(0, {"target": _usage(100), "optimizer": _usage(0)})
    data = load_run(run.path)
    assert data["usage"]["target"] == _usage(913)
    assert data["usage"]["optimizer"] is None
    assert data["rounds"][0]["usage"] == {"target": _usage(100), "optimizer": _usage(0)}
    assert data["rounds"][1]["usage"] is None


# --- AC21: write_report -----------------------------------------------------------------

def test_write_report_creates_file(tmp_path):
    run = SynRun(tmp_path)
    run.skill(0, "a")
    out = write_report(run.path)
    assert Path(out) == run.path / "report.html"
    assert (run.path / "report.html").is_file()
    assert not (run.path / "report.html.tmp").exists()


def test_write_report_accepts_str(tmp_path):
    run = SynRun(tmp_path)
    run.skill(0, "a")
    assert Path(write_report(str(run.path))) == run.path / "report.html"


def test_write_report_overwrites(tmp_path):
    run = SynRun(tmp_path)
    run.skill(0, "a", instruction="First version text.")
    write_report(run.path)
    assert "First version text." in (run.path / "report.html").read_text(encoding="utf-8")
    run.skill(0, "zz", instruction="Second version text.")
    write_report(run.path)
    text = (run.path / "report.html").read_text(encoding="utf-8")
    assert "Second version text." in text
    assert not (run.path / "report.html.tmp").exists()


def test_write_report_utf8(tmp_path):
    run = SynRun(tmp_path)
    run.skill(0, "a", instruction="단계별로 짧게 생각하라.")
    write_report(run.path)
    assert "단계별로 짧게 생각하라." in (run.path / "report.html").read_text(encoding="utf-8")



# --- HTML helpers (redesigned layout) ---------------------------------------------------

_OP_CLASS = re.compile(r"^op-(c\d+|none)$")


def _text(el) -> str:
    """Visible text of an element (tags stripped, entities unescaped, whitespace collapsed)."""
    return " ".join(html_lib.unescape(re.sub(r"<[^>]*>", " ", el.html)).split())


def _nodes(html):
    return find_elements(html, cls="node", tag="g")


def _node(html, key):
    found = [n for n in _nodes(html) if n.attrs.get("data-node") == key]
    assert len(found) == 1, (key, len(found))
    return found[0]


def _op_class(el) -> str:
    ops = [c for c in el.classes if _OP_CLASS.match(c)]
    assert len(ops) == 1, (el.attrs, el.classes)
    return ops[0]


def _slots(html) -> dict:
    return {n.attrs["data-node"]: int(n.attrs["data-slot"]) for n in _nodes(html)}


def _edges(html) -> list:
    out = []
    for e in find_elements(html, cls="edge", tag="path"):
        kind = "primary" if "primary" in e.classes else ("secondary" if "secondary" in e.classes else None)
        assert not ("primary" in e.classes and "secondary" in e.classes), e.attrs
        out.append((e.attrs.get("data-src"), e.attrs.get("data-dst"), kind))
    return out


def _article(html, key):
    art = element_by_id(html, key)
    assert art is not None, key
    assert art.tag == "article", (key, art.tag)
    return art


def _card(html, name):
    cards = [c for c in find_elements(element_by_id(html, "summary").html, cls="card", tag="div")
             if c.attrs.get("data-card") == name]
    assert len(cards) == 1, (name, len(cards))
    return cards[0]


def _script(html) -> str:
    m = re.search(r"<script[^>]*>(.*?)</script>", html, re.DOTALL | re.IGNORECASE)
    assert m is not None
    return m.group(1)


def _body(html):
    bodies = find_elements(html, tag="body")
    assert len(bodies) == 1
    return bodies[0]


def _render(run) -> str:
    return render_report(load_run(run.path))


LINEAGE_KEYS = ["r000-a", "r000-x", "r001-a", "r001-b", "r001-b2", "r002-c", "r002-d", "r002-e",
                "r002-g", "r002-h", "r002-k"]


# --- AC3: refresh -------------------------------------------------------------------------

def test_refresh_only_while_running(tmp_path):
    run = SynRun(tmp_path)
    run.skill(0, "a")
    write_report(run.path)
    assert REFRESH_META in (run.path / "report.html").read_text(encoding="utf-8")
    run.finish()
    write_report(run.path)
    assert 'http-equiv="refresh"' not in (run.path / "report.html").read_text(encoding="utf-8")


def test_refresh_meta_in_noscript_and_body_attr_running(tmp_path):
    run = SynRun(tmp_path, name="20261116-000000_live")
    run.skill(0, "a")
    html = _render(run)
    noscripts = find_elements(html, tag="noscript")
    assert any(REFRESH_META in n.html for n in noscripts)
    body = _body(html)
    assert body.attrs.get("data-refresh-seconds") == "5"
    assert body.attrs.get("data-run") == "20261116-000000_live"


def test_refresh_absent_when_finished(tmp_path):
    run = SynRun(tmp_path, name="20261116-000000_done")
    run.skill(0, "a").finish()
    html = _render(run)
    assert 'http-equiv="refresh"' not in html
    body = _body(html)
    assert "data-refresh-seconds" not in body.attrs
    assert body.attrs.get("data-run") == "20261116-000000_done"


def test_running_run_without_rounds_renders(tmp_path):
    run = SynRun(tmp_path)
    html = _render(run)
    assert html.lstrip().startswith("<!DOCTYPE html>")
    assert REFRESH_META in html
    assert _nodes(html) == []
    assert find_elements(html, cls="pt", tag="circle") == []
    assert find_elements(html, cls="row", tag="tr") == []


# --- AC4: self-contained ------------------------------------------------------------------

@pytest.mark.parametrize("finished", [False, True])
def test_self_contained(tmp_path, finished):
    run = _lineage_run(tmp_path)
    if finished:
        run.finish()
    html = _render(run)
    assert html.lstrip().startswith("<!DOCTYPE html>")
    for bad in ('src="http', 'href="http', "<script src", '<link rel="stylesheet"'):
        assert bad not in html, bad
    assert len(find_elements(html, tag="style")) == 1
    scripts = find_elements(html, tag="script")
    assert len(scripts) == 1
    assert "src" not in scripts[0].attrs


# --- AC5: sections, header ------------------------------------------------------------------

def test_sections_in_order(tmp_path):
    html = _render(_lineage_run(tmp_path).finish())
    starts = []
    for sid, tag in (("header", "header"), ("evolution", "section"), ("summary", "section"),
                     ("trends", "section"), ("table", "section")):
        el = element_by_id(html, sid)
        assert el is not None, sid
        assert el.tag == tag, (sid, el.tag)
        starts.append(el.start)
    assert starts == sorted(starts)


def test_header_run_and_optimizer(tmp_path):
    run = SynRun(tmp_path, name="20261116-000000_syn-alpha", optimizer="my_opt")
    run.skill(0, "a")
    header = _text(element_by_id(_render(run), "header"))
    assert "20261116-000000_syn-alpha" in header
    assert "my_opt" in header


# --- AC6: tree nodes ------------------------------------------------------------------------

def test_tree_nodes(tmp_path):
    html = _render(_lineage_run(tmp_path))
    tree = element_by_id(html, "tree")
    assert tree is not None and tree.tag == "svg"
    vp = element_by_id(html, "viewport")
    assert vp is not None and vp.tag == "g"
    assert tree.start < vp.start and vp.end <= tree.end
    evo = element_by_id(html, "evolution")
    assert evo.start < tree.start and tree.end <= evo.end
    assert find_elements(vp.html, cls="edges", tag="g")
    assert find_elements(vp.html, cls="nodes", tag="g")
    nodes = _nodes(html)
    assert sorted(n.attrs["data-node"] for n in nodes) == sorted(LINEAGE_KEYS)
    assert len(_nodes(vp.html)) == len(LINEAGE_KEYS)
    ops = {"r000-a": "initial", "r001-a": "reeval", "r002-c": "crossover", "r002-d": "", "r002-g": "mutate"}
    for n in nodes:
        key = n.attrs["data-node"]
        rnd, sid = int(key[1:4]), key[5:]
        assert n.attrs.get("data-round") == str(rnd), key
        assert n.attrs.get("data-skill") == sid, key
        assert "data-op" in n.attrs, key
        assert "data-slot" in n.attrs and n.attrs["data-slot"].isdigit(), key
        assert n.attrs.get("tabindex") == "0", key
        if key in ops:
            assert n.attrs["data-op"] == ops[key], key
        titles = find_elements(n.html, tag="title")
        assert any(sid in _text(t) for t in titles), key
        ids = find_elements(n.html, cls="node-id", tag="text")
        assert len(ids) == 1, key


def test_round_labels(tmp_path):
    run = SynRun(tmp_path)
    run.skill(0, "a").skill(1, "b").skill(10, "c")
    (run.path / "rounds" / "002" / "skills").mkdir(parents=True)  # no summary.json
    run.raw("rounds/003/summary.json", "{not json")
    html = _render(run)
    assert len(find_elements(html, cls="round-label", tag="text")) == 3
    assert sorted(n.attrs["data-node"] for n in _nodes(html)) == ["r000-a", "r001-b", "r010-c"]


def test_round_zero_only_tree(tmp_path):
    run = SynRun(tmp_path)
    run.skill(0, "a")
    html = _render(run)
    assert [n.attrs["data-node"] for n in _nodes(html)] == ["r000-a"]
    assert len(find_elements(html, cls="round-label", tag="text")) == 1
    assert _edges(html) == []


# --- AC7: node bars -------------------------------------------------------------------------

def test_node_bars(tmp_path):
    run = SynRun(tmp_path)
    run.skill(0, "a", benches={"gsm8k": (0.5, 10.0), "bbh": (0.25, 20.0)})
    run.skill(0, "b", benches={"gsm8k": (0.75, 30.0)})
    html = _render(run)
    a = _node(html, "r000-a")
    bars = find_elements(a.html, cls="bar", tag="g")
    assert [b.attrs.get("data-benchmark") for b in bars] == ["gsm8k", "bbh"]
    assert [float(b.attrs["data-value"]) for b in bars] == [0.5, 0.25]
    assert [_text(t) for t in find_elements(a.html, cls="tok", tag="text")] == ["10", "20"]
    b = _node(html, "r000-b")
    bars = find_elements(b.html, cls="bar", tag="g")
    assert [x.attrs.get("data-benchmark") for x in bars] == ["gsm8k", "bbh"]
    assert float(bars[0].attrs["data-value"]) == 0.75
    assert bars[1].attrs.get("data-value") == ""
    assert [_text(t) for t in find_elements(b.html, cls="tok", tag="text")] == ["30", "-"]


def test_node_bar_null_accuracy(tmp_path):
    run = SynRun(tmp_path)
    run.skill(0, "a", benches={"gsm8k": (None, 10.0)})
    bars = find_elements(_node(_render(run), "r000-a").html, cls="bar", tag="g")
    assert len(bars) == 1
    assert bars[0].attrs.get("data-value") == ""


# --- AC8: edges ----------------------------------------------------------------------------

def test_edges_lineage(tmp_path):
    html = _render(_lineage_run(tmp_path))
    got = _edges(html)
    expected = [
        ("r000-a", "r001-b", "primary"),
        ("r000-x", "r001-b2", "primary"),
        ("r001-a", "r002-c", "primary"),
        ("r000-x", "r002-c", "secondary"),
        ("r001-b", "r002-d", "primary"),  # parent_id fallback
        ("r000-x", "r002-k", "primary"),
    ]
    assert sorted(got) == sorted(expected)
    # unresolved (ghost) and same-round (h <- c) parents draw nothing
    assert not any(dst in ("r002-g", "r002-h") for _, dst, _ in got)


def test_edges_duplicate_parent_ids(tmp_path):
    run = SynRun(tmp_path)
    run.skill(0, "a").skill(0, "b")
    run.skill(1, "c", origin=_origin(["a", "a"], "mutate"))
    run.skill(1, "d", origin=_origin(["a", "b", "a"], "crossover"))
    got = _edges(_render(run))
    assert sorted(got) == sorted([
        ("r000-a", "r001-c", "primary"),
        ("r000-a", "r001-d", "primary"),
        ("r000-b", "r001-d", "secondary"),
    ])


def test_edges_unresolved_first_parent(tmp_path):
    run = SynRun(tmp_path)
    run.skill(0, "a")
    run.skill(1, "c", origin=_origin(["ghost", "a"], "crossover"))
    assert _edges(_render(run)) == [("r000-a", "r001-c", "secondary")]


# --- AC9: slot layout ------------------------------------------------------------------------

def _slot_run(root, accs=None):
    accs = accs or {}

    def b(sid):
        return {"gsm8k": (accs.get(sid, 0.5), 10.0)}

    run = SynRun(root)
    for sid in ("zz", "aa", "mm"):  # round 0: saved order kept (slots 0, 1, 2)
        run.skill(0, sid, benches=b(sid), origin=_origin([], "initial"))
    # round 1 saved order differs from parent order
    run.skill(1, "n-none", benches=b("n-none"))
    run.skill(1, "c-mm", benches=b("c-mm"), origin=_origin(["mm"]))
    run.skill(1, "c-zz", benches=b("c-zz"), origin=_origin(["zz"]))
    run.skill(1, "c-aa1", benches=b("c-aa1"), origin=_origin(["aa"]))
    run.skill(1, "c-aa2", benches=b("c-aa2"), origin=_origin(["aa"]))
    run.skill(1, "c-ghost-zz", benches=b("c-ghost-zz"), origin=_origin(["ghost", "zz"], "crossover"))
    run.skill(1, "n-ghost", benches=b("n-ghost"), origin=_origin(["ghost"]))
    # round 2: same parent slot, different parent rounds
    run.skill(2, "d-none", benches=b("d-none"), origin=_origin(["n-none"]))   # (0, 5, 1)
    run.skill(2, "d-aa1", benches=b("d-aa1"), origin=_origin(["c-aa1"]))      # (0, 2, 1)
    run.skill(2, "d-mm", benches=b("d-mm"), origin=_origin(["mm"]))           # (0, 2, 0)
    run.skill(2, "d-zz", benches=b("d-zz"), origin=_origin(["zz"]))           # (0, 0, 0)
    return run


SLOT_EXPECTED = {
    "r000-zz": 0, "r000-aa": 1, "r000-mm": 2,
    "r001-c-zz": 0, "r001-c-ghost-zz": 1, "r001-c-aa1": 2, "r001-c-aa2": 3, "r001-c-mm": 4,
    "r001-n-none": 5, "r001-n-ghost": 6,
    "r002-d-zz": 0, "r002-d-mm": 1, "r002-d-aa1": 2, "r002-d-none": 3,
}


def test_slot_layout(tmp_path):
    assert _slots(_render(_slot_run(tmp_path))) == SLOT_EXPECTED


def test_slot_layout_ignores_accuracy(tmp_path):
    base = _slots(_render(_slot_run(tmp_path / "one")))
    accs = {"zz": 0.0, "aa": 1.0, "mm": 0.75, "n-none": 1.0, "c-mm": 0.0, "c-zz": 0.1, "c-aa2": 1.0,
            "n-ghost": 0.9, "d-none": 1.0, "d-zz": 0.0}
    assert _slots(_render(_slot_run(tmp_path / "two", accs))) == base


# --- AC10: operation colors ----------------------------------------------------------------

def test_op_classes_lineage(tmp_path):
    html = _render(_lineage_run(tmp_path))
    cls = {n.attrs["data-node"]: _op_class(n) for n in _nodes(html)}
    assert cls["r000-a"] == cls["r000-x"] == "op-c0"     # initial
    assert cls["r001-a"] == "op-c1"                      # reeval
    assert cls["r001-b"] == cls["r001-b2"] == "op-c2"    # rewrite
    assert cls["r002-c"] == "op-c3"                      # crossover
    assert cls["r002-d"] == cls["r002-e"] == "op-none"   # no origin
    assert cls["r002-g"] == cls["r002-h"] == cls["r002-k"]
    assert cls["r002-g"] not in {"op-c0", "op-c1", "op-c2", "op-c3", "op-none"}


def test_legend(tmp_path):
    html = _render(_lineage_run(tmp_path))
    legend = element_by_id(html, "legend")
    assert legend is not None and legend.tag == "ul"
    evo = element_by_id(html, "evolution")
    assert evo.start < legend.start and legend.end <= evo.end
    items = [li for li in find_elements(legend.html, tag="li") if "data-op" in li.attrs]
    assert [li.attrs["data-op"] for li in items] == ["initial", "reeval", "rewrite", "crossover", "", "mutate"]
    node_cls = {}
    for n in _nodes(html):
        node_cls.setdefault(n.attrs["data-op"], _op_class(n))
    for li in items:
        assert _op_class(li) == node_cls[li.attrs["data-op"]], li.attrs
    unknown = [li for li in items if li.attrs["data-op"] == ""][0]
    assert "unknown" in _text(unknown)
    assert _op_class(unknown) == "op-none"


def test_op_classes_cycle(tmp_path):
    run = SynRun(tmp_path)
    for i in range(10):
        run.skill(0, f"s{i}", origin=_origin([], f"op{i}"))
    html = _render(run)
    cls = [_op_class(_node(html, f"r000-s{i}")) for i in range(10)]
    assert cls == [f"op-c{k}" for k in range(8)] + ["op-c0", "op-c1"]
    items = [li for li in find_elements(element_by_id(html, "legend").html, tag="li") if "data-op" in li.attrs]
    assert [li.attrs["data-op"] for li in items] == [f"op{i}" for i in range(10)]
    assert [_op_class(li) for li in items] == cls


def test_op_classes_ignore_accuracy(tmp_path):
    def build(root, acc_a, acc_b):
        run = SynRun(root)
        run.skill(0, "a", benches={"gsm8k": (acc_a, 10.0)}, origin=_origin([], "initial"))
        run.skill(0, "b", benches={"gsm8k": (acc_b, 10.0)}, origin=_origin([], "seed"))
        run.skill(1, "c", benches={"gsm8k": (acc_a, 10.0)}, origin=_origin(["a"], "rewrite"))
        run.skill(1, "d", benches={"gsm8k": (acc_b, 10.0)}, origin=_origin(["b"], "mutate"))
        html = _render(run)
        return ({n.attrs["data-node"]: (_op_class(n), n.attrs["data-slot"]) for n in _nodes(html)},
                [(_op_class(li), li.attrs["data-op"]) for li in find_elements(element_by_id(html, "legend").html,
                                                                              tag="li") if "data-op" in li.attrs])

    assert build(tmp_path / "one", 0.25, 1.0) == build(tmp_path / "two", 1.0, 0.25)


# --- AC11: detail panel ---------------------------------------------------------------------

def test_detail_panel_articles(tmp_path):
    html = _render(_lineage_run(tmp_path))
    detail = element_by_id(html, "detail")
    assert detail is not None and detail.tag == "aside"
    evo = element_by_id(html, "evolution")
    assert evo.start < detail.start and detail.end <= evo.end
    empty = element_by_id(detail.html, "detail-empty")
    assert empty is not None and empty.tag == "div"
    arts = find_elements(detail.html, cls="skill", tag="article")
    assert sorted(a.attrs.get("id") for a in arts) == sorted(LINEAGE_KEYS)
    for a in arts:
        assert a.attrs.get("data-node") == a.attrs.get("id")
        assert "hidden" in a.attrs, a.attrs


def test_skill_size_span(tmp_path):
    run = SynRun(tmp_path)
    run.skill(0, "a", num_tokens=42)
    run.skill(0, "b", num_tokens=None)
    html = _render(run)
    assert '<span class="size">42 tokens</span>' in _article(html, "r000-a").html
    assert '<span class="size">-</span>' in _article(html, "r000-b").html


def test_skill_size_missing_skill_json(tmp_path):
    run = SynRun(tmp_path)
    run.skill(0, "a", skill_json=False)
    assert '<span class="size">-</span>' in _article(_render(run), "r000-a").html


def test_detail_op_and_note(tmp_path):
    run = SynRun(tmp_path)
    run.skill(0, "a", origin=_origin([], "initial", "first draft note"))
    run.skill(0, "b")  # no origin
    run.skill(0, "c", origin=_origin([], "seed", ""))
    html = _render(run)
    a = _article(html, "r000-a").html
    ops = find_elements(a, cls="op", tag="span")
    assert len(ops) == 1 and "initial" in _text(ops[0])
    notes = find_elements(a, cls="note", tag="div")
    assert len(notes) == 1 and "first draft note" in _text(notes[0])
    b = _article(html, "r000-b").html
    assert find_elements(b, cls="op", tag="span") == []
    assert find_elements(b, cls="note", tag="div") == []
    assert find_elements(_article(html, "r000-c").html, cls="note", tag="div") == []


def test_detail_parent_links(tmp_path):
    html = _render(_lineage_run(tmp_path))

    def parents(key):
        found = find_elements(_article(html, key).html, cls="parents", tag="div")
        assert len(found) == 1, key
        return found[0]

    assert 'href="#r000-a"' in parents("r001-b").html
    c = parents("r002-c").html
    assert 'href="#r001-a"' in c and 'href="#r000-x"' in c
    assert 'href="#r000-a"' not in c
    assert 'href="#r001-b"' in parents("r002-d").html
    assert "-" in _text(parents("r000-a"))
    assert find_elements(parents("r000-a").html, tag="a") == []


def test_detail_unresolved_parent(tmp_path):
    html = _render(_lineage_run(tmp_path))
    for key, pid in (("r002-g", "ghost"), ("r002-h", "c")):
        div = find_elements(_article(html, key).html, cls="parents", tag="div")[0]
        unresolved = [el for el in find_elements(div.html, cls="unresolved", tag="span") if pid in _text(el)]
        assert unresolved, key
        assert all('href="#r' not in el.html for el in unresolved), key
        assert 'href="#r' not in div.html, key


def test_detail_children(tmp_path):
    html = _render(_lineage_run(tmp_path))

    def kids(key):
        return "".join(el.html for el in find_elements(_article(html, key).html, cls="children", tag="div"))

    x = kids("r000-x")
    for href in ('href="#r001-b2"', 'href="#r002-c"', 'href="#r002-k"'):
        assert href in x, href
    a0 = kids("r000-a")
    assert 'href="#r001-b"' in a0 and 'href="#r002-c"' not in a0
    assert 'href="#r002-c"' in kids("r001-a")
    assert kids("r002-e") == ""
    assert kids("r001-b2") == ""


def test_detail_instruction_and_few_shot(tmp_path):
    instr = 'Line one & "two".\nLine <three> step by step.'
    run = SynRun(tmp_path)
    run.skill(0, "a", instruction=instr, few_shot=[{"question": "1+1?", "answer": "2"}])
    run.skill(0, "b", instruction="Plain.")
    html = _render(run)
    a = _article(html, "r000-a").html
    pres = find_elements(a, cls="instruction", tag="pre")
    assert len(pres) == 1
    assert html_lib.unescape(re.sub(r"<[^>]*>", "", pres[0].html)).strip() == instr
    assert len(find_elements(a, cls="few-shot", tag="details")) == 1
    assert "1+1?" in a
    assert find_elements(_article(html, "r000-b").html, cls="few-shot", tag="details") == []


# --- AC12: diff -------------------------------------------------------------------------------

def _diff_html(html, key):
    return "".join(el.html for el in find_elements(_article(html, key).html, cls="diff", tag="div"))


def test_diff_insert(tmp_path):
    diff = _diff_html(_render(_lineage_run(tmp_path)), "r001-b")
    assert diff
    assert "<ins>briefly</ins>" in diff


def test_diff_delete(tmp_path):
    run = SynRun(tmp_path)
    run.skill(0, "p", instruction="Think briefly step by step.")
    run.skill(1, "q", instruction="Think step by step.", origin=_origin(["p"], "rewrite"))
    diff = _diff_html(_render(run), "r001-q")
    assert re.search(r"<del>\s*briefly\s*</del>", diff), diff


def test_diff_against_first_parent(tmp_path):
    run = SynRun(tmp_path)
    run.skill(0, "p1", instruction="Think step by step.")
    run.skill(0, "p2", instruction="Answer directly.")
    run.skill(1, "q", instruction="Think carefully step by step.", origin=_origin(["p1", "p2"], "crossover"))
    html = _render(run)
    q = _article(html, "r001-q").html
    assert 'href="#r000-p1"' in q and 'href="#r000-p2"' in q
    diff = _diff_html(html, "r001-q")
    assert "<ins>carefully</ins>" in diff
    assert "directly" not in diff


def test_no_diff_without_resolved_first_parent(tmp_path):
    html = _render(_lineage_run(tmp_path))
    for key in ("r000-a", "r002-e", "r002-g", "r002-h"):
        assert _diff_html(html, key) == "", key
    run = SynRun(tmp_path / "second")
    run.skill(0, "a", instruction="Think.")
    run.skill(1, "c", instruction="Think more.", origin=_origin(["ghost", "a"], "crossover"))
    assert _diff_html(_render(run), "r001-c") == ""


# --- AC13: metrics -----------------------------------------------------------------------------

def test_detail_metrics(tmp_path):
    run = SynRun(tmp_path)
    run.skill(0, "a", benches={"gsm8k": (0.5, 10.0), "bbh": (0.25, 20.0)})
    run.skill(0, "b", benches={"gsm8k": (0.75, 30.0)})
    html = _render(run)
    for key, expected in (("r000-a", ["gsm8k", "bbh"]), ("r000-b", ["gsm8k"])):
        tables = find_elements(_article(html, key).html, cls="metrics", tag="table")
        assert len(tables) == 1, key
        rows = [tr.attrs["data-benchmark"] for tr in find_elements(tables[0].html, tag="tr")
                if "data-benchmark" in tr.attrs]
        assert rows == expected, key


# --- AC14: failures ----------------------------------------------------------------------------

def _fail_rows(sid, bench, k, n_ok=1):
    rows = [_item(sid, bench, i, True, question=f"OKQ-{sid}-{bench}-{i:03d}") for i in range(n_ok)]
    rows += [_item(sid, bench, n_ok + i, False, question=f"FAILQ-{sid}-{bench}-{i:03d}") for i in range(k)]
    return rows


def test_detail_failures(tmp_path):
    run = SynRun(tmp_path)
    run.skill(0, "a", benches={"gsm8k": (0.5, 1.0), "bbh": (0.5, 1.0)}, fails={"gsm8k": 25, "bbh": 0},
              items={"gsm8k": _fail_rows("a", "gsm8k", 25), "bbh": _fail_rows("a", "bbh", 0, n_ok=4)})
    run.skill(0, "b", benches={"gsm8k": (0.5, 1.0), "bbh": (0.5, 1.0)}, fails={"gsm8k": 0, "bbh": 3},
              items={"gsm8k": _fail_rows("b", "gsm8k", 0, n_ok=4), "bbh": _fail_rows("b", "bbh", 3)})
    run.skill(0, "c", benches={"gsm8k": (0.5, 1.0)}, fails={"gsm8k": 4},
              items={"gsm8k": _fail_rows("c", "gsm8k", 4)})
    html = _render(run)

    def fails(key):
        return {el.attrs.get("data-benchmark"): el
                for el in find_elements(_article(html, key).html, cls="failures", tag="details")}

    fa = fails("r000-a")
    assert set(fa) == {"gsm8k"}
    g = fa["gsm8k"]
    summ = find_elements(g.html, tag="summary")
    assert summ and "gsm8k: 25 incorrect" in _text(summ[0])
    shown = re.findall(r"FAILQ-a-gsm8k-(\d{3})", g.html)
    assert sorted(set(shown)) == ["000", "001", "002"]
    assert "+22 more" in g.html
    assert "The answer is 2." in g.html

    fb = fails("r000-b")
    assert set(fb) == {"bbh"}
    assert "bbh: 3 incorrect" in _text(find_elements(fb["bbh"].html, tag="summary")[0])
    assert sorted(set(re.findall(r"FAILQ-b-bbh-(\d{3})", fb["bbh"].html))) == ["000", "001", "002"]
    assert not re.search(r"\+\s*\d+\s*more", fb["bbh"].html)

    fc = fails("r000-c")
    assert "+1 more" in fc["gsm8k"].html
    assert sorted(set(re.findall(r"FAILQ-c-gsm8k-(\d{3})", fc["gsm8k"].html))) == ["000", "001", "002"]

    assert "+0 more" not in html
    table = element_by_id(html, "table").html
    assert find_elements(table, cls="failures") == []
    assert "FAILQ-" not in table


def test_long_question_truncated(tmp_path):
    long_q = "".join(f"{i:04d}" for i in range(250))
    assert len(long_q) == 1000
    run = SynRun(tmp_path)
    run.skill(0, "a", items={"gsm8k": [_item("a", "gsm8k", 0, False, question=long_q)]})
    html = _render(run)
    assert long_q not in html
    assert long_q[:MAX_TEXT_CHARS + 1] not in html
    assert long_q[:MAX_TEXT_CHARS] + "..." in html


# --- AC15: summary cards -------------------------------------------------------------------------

def test_summary_cards_order(tmp_path):
    html = _render(_lineage_run(tmp_path))
    cards = [c.attrs.get("data-card") for c in find_elements(element_by_id(html, "summary").html,
                                                               cls="card", tag="div") if "data-card" in c.attrs]
    assert cards == ["round", "status", "skills", "target-usage", "optimizer-usage", "scoring"]


def test_summary_running(tmp_path):
    run = _lineage_run(tmp_path)
    run.usage("target", _usage(1234567))
    html = _render(run)
    assert "running" in _text(_card(html, "status"))
    assert re.search(r"2\s*/\s*3", _text(_card(html, "round"))), _text(_card(html, "round"))
    skills = _text(_card(html, "skills"))
    assert re.search(r"\b11\b", skills) and re.search(r"\b10\b", skills), skills
    assert "1,234,567" in _text(_card(html, "target-usage"))
    opt = _text(_card(html, "optimizer-usage"))
    assert "-" in opt
    assert "1,234,567" not in opt


def test_summary_finished(tmp_path):
    run = SynRun(tmp_path, optimizer="opro")
    run.skill(0, "a")
    run.usage("target", _usage(913)).usage("optimizer", _usage(457))
    run.finish(stop_reason="empty_update")
    html = _render(run)
    assert "finished: empty_update" in _text(_card(html, "status"))
    assert "running" not in _text(_card(html, "status"))
    assert "913" in _text(_card(html, "target-usage"))
    assert "457" in _text(_card(html, "optimizer-usage"))


def test_summary_usage_missing(tmp_path):
    run = SynRun(tmp_path)
    run.skill(0, "a")
    html = _render(run)
    for name in ("target-usage", "optimizer-usage"):
        assert "-" in _text(_card(html, name)), name


# --- AC16: trends ---------------------------------------------------------------------------------

def _trend_run(root):
    run = SynRun(root)
    run.skill(0, "a", benches={"gsm8k": (0.5, 10.0), "bbh": (0.25, 20.0)})
    run.skill(0, "b", benches={"gsm8k": (0.75, 30.0)})  # gsm8k only
    run.skill(1, "c", benches={"gsm8k": (1.0, 5.0), "bbh": (0.5, 8.0)})
    return run


def _bench_block(html, key):
    blocks = [el for el in find_elements(element_by_id(html, "trends").html, cls="bench", tag="div")
              if el.attrs.get("data-benchmark") == key]
    assert len(blocks) == 1, (key, len(blocks))
    return blocks[0]


def _chart(block, metric):
    charts = [c for c in find_elements(block.html, cls="chart", tag="svg") if c.attrs.get("data-metric") == metric]
    assert len(charts) == 1, metric
    return charts[0]


def test_trends_blocks(tmp_path):
    html = _render(_trend_run(tmp_path))
    blocks = [el.attrs.get("data-benchmark") for el in find_elements(element_by_id(html, "trends").html,
                                                                     cls="bench", tag="div")]
    assert blocks == ["gsm8k", "bbh"]
    node_cls = {n.attrs["data-node"]: _op_class(n) for n in _nodes(html)}
    expected = {"gsm8k": [(0, "a"), (0, "b"), (1, "c")], "bbh": [(0, "a"), (1, "c")]}
    for key, pts in expected.items():
        block = _bench_block(html, key)
        assert len(find_elements(block.html, tag="svg")) == 2, key
        for metric in ("accuracy", "avg_output_tokens"):
            circles = find_elements(_chart(block, metric).html, cls="pt", tag="circle")
            got = sorted((int(c.attrs["data-round"]), c.attrs["data-skill"]) for c in circles)
            assert got == sorted(pts), (key, metric)
            for c in circles:
                nkey = f"r{int(c.attrs['data-round']):03d}-{c.attrs['data-skill']}"
                assert c.attrs.get("data-node") == nkey
                assert _op_class(c) == node_cls[nkey]
                titles = find_elements(c.html, tag="title")
                assert titles and c.attrs["data-skill"] in _text(titles[0])


def test_trends_null_accuracy_no_point(tmp_path):
    run = SynRun(tmp_path)
    run.skill(0, "a", benches={"gsm8k": (None, 10.0)})
    run.skill(0, "b", benches={"gsm8k": (0.5, 12.0)})
    block = _bench_block(_render(run), "gsm8k")
    acc = [c.attrs["data-skill"] for c in find_elements(_chart(block, "accuracy").html, cls="pt", tag="circle")]
    tok = [c.attrs["data-skill"] for c in
           find_elements(_chart(block, "avg_output_tokens").html, cls="pt", tag="circle")]
    assert acc == ["b"]
    assert sorted(tok) == ["a", "b"]


def test_trends_zero_tokens(tmp_path):
    run = SynRun(tmp_path)
    run.skill(0, "a", benches={"gsm8k": (0.0, 0.0)})
    run.skill(0, "b", benches={"gsm8k": (0.0, 0.0)})
    block = _bench_block(_render(run), "gsm8k")
    assert len(find_elements(_chart(block, "accuracy").html, cls="pt", tag="circle")) == 2
    assert len(find_elements(_chart(block, "avg_output_tokens").html, cls="pt", tag="circle")) == 2


# --- AC17: table ----------------------------------------------------------------------------------

def test_table_collapsed(tmp_path):
    html = _render(_lineage_run(tmp_path))
    det = element_by_id(html, "table-details")
    assert det is not None and det.tag == "details"
    assert "open" not in det.attrs
    table = element_by_id(det.html, "skill-table")
    assert table is not None and table.tag == "table"
    sec = element_by_id(html, "table")
    assert sec.start < det.start and det.end <= sec.end


def test_table_rows_in_saved_order(tmp_path):
    run = _lineage_run(tmp_path)
    html = _render(run)
    rows = find_elements(element_by_id(html, "skill-table").html, cls="row", tag="tr")
    got = [(int(r.attrs["data-round"]), r.attrs["data-skill"]) for r in rows]
    expected = [(rnd, e["skill_id"]) for rnd in sorted(run.summaries) for e in run.summaries[rnd]]
    assert got == expected
    assert [r.attrs.get("data-order") for r in rows] == [str(i) for i in range(len(rows))]
    for r, (rnd, sid) in zip(rows, expected):
        key = f"r{rnd:03d}-{sid}"
        assert r.attrs.get("data-node") == key
        assert f'href="#{key}"' in r.html


def test_table_sort_headers(tmp_path):
    html = _render(_trend_run(tmp_path))
    table = element_by_id(html, "skill-table").html
    heads = [el for el in html_elements(table) if "data-sort-col" in el.attrs]
    assert heads
    cols = [el.attrs["data-sort-col"] for el in heads]
    assert all(c.isdigit() for c in cols)
    assert len(set(cols)) == len(cols)
    types = {el.attrs.get("data-sort-type") for el in heads}
    assert types <= {"num", "text"}
    assert types == {"num", "text"}


def test_table_numeric_cells(tmp_path):
    run = SynRun(tmp_path)
    ini = _origin([], "initial")
    run.skill(0, "a", benches={"gsm8k": (0.5, 1.0), "bbh": (0.25, 2.0)}, num_tokens=42, origin=ini)
    run.skill(0, "b", benches={"gsm8k": (0.75, 1.0)}, num_tokens=42, origin=ini)
    run.skill(0, "c", benches={"gsm8k": (0.5, 1.0), "bbh": (0.5, 1.0)}, num_tokens=None, origin=ini)
    html = _render(run)
    rows = {r.attrs["data-skill"]: r for r in find_elements(element_by_id(html, "skill-table").html,
                                                             cls="row", tag="tr")}
    assert set(rows) == {"a", "b", "c"}

    def cells(sid):
        return [td for td in find_elements(rows[sid].html, tag="td") if "data-value" in td.attrs]

    def num(v):
        try:
            return float(v)
        except ValueError:
            return None

    for sid in rows:
        for td in cells(sid):
            if td.attrs["data-value"] == "":
                assert _text(td) == "-", (sid, td.html)
    a_vals = [td.attrs["data-value"] for td in cells("a")]
    assert "" not in a_vals
    assert 0.5 in [num(v) for v in a_vals] and 0.25 in [num(v) for v in a_vals]
    assert 42.0 in [num(v) for v in a_vals]
    # b misses bbh: acc, in, out, think, trunc rate -> empty
    assert [td.attrs["data-value"] for td in cells("b")].count("") >= 5
    # c: only the size is missing
    c_vals = [td.attrs["data-value"] for td in cells("c")]
    assert c_vals.count("") == 1
    assert 42.0 not in [num(v) for v in c_vals]


# --- AC18: no judgment ----------------------------------------------------------------------------

def test_no_judgment_words(tmp_path):
    lineage = _lineage_run(tmp_path / "one").finish()
    lineage.usage("target", _usage(913))
    trend = _trend_run(tmp_path / "two")
    fails = SynRun(tmp_path / "three")
    fails.skill(0, "a", fails={"gsm8k": 5})
    empty = SynRun(tmp_path / "four")
    for run in (lineage, trend, fails, empty):
        low = _render(run).lower()
        assert "best" not in low
        assert "winner" not in low


# --- AC19: escaping --------------------------------------------------------------------------------

EVIL = "<script>alert(1)</script>"
QUOTEY = "q\"u'o&te"


def test_escaping(tmp_path):
    run = SynRun(tmp_path)
    run.skill(0, "p", instruction="Plain.")
    run.skill(1, "a", instruction=f"Do {EVIL} it {QUOTEY}.", origin=_origin(["p"], f"op{EVIL}", f"note {EVIL} {QUOTEY}"),
              items={"gsm8k": [_item("a", "gsm8k", 0, False, question=f"Q {EVIL} {QUOTEY}?")]})
    html = _render(run)
    assert EVIL not in html
    assert QUOTEY not in html
    assert "&lt;script&gt;" in html
    assert "&amp;te" in html


def test_escaping_ids_and_benchmarks(tmp_path):
    run = SynRun(tmp_path, name="20261116-000000_a&b")
    run.skill(0, "p")
    sid = f"id{EVIL}{QUOTEY}"
    bench = f"bench{EVIL}{QUOTEY}"
    run.skill(0, sid, benches={bench: (0.5, 10.0)}, skill_json=False, jsonl=False)
    run.skill(1, "c", origin=_origin([sid], f"op{QUOTEY}"))
    html = _render(run)
    assert EVIL not in html
    assert QUOTEY not in html
    assert "_a&b" not in html
    assert _body(html).attrs.get("data-run") == "20261116-000000_a&b"
    node = _node(html, f"r000-{sid}")
    assert node.attrs.get("data-skill") == sid
    assert bench in [b.attrs.get("data-benchmark") for b in find_elements(node.html, cls="bar", tag="g")]
    assert [b.attrs.get("data-benchmark") for b in find_elements(element_by_id(html, "trends").html,
                                                                 cls="bench", tag="div")] == ["gsm8k", bench]
    assert _node(html, "r001-c").attrs.get("data-op") == f"op{QUOTEY}"


# --- AC20: script sanity ---------------------------------------------------------------------------

@pytest.mark.parametrize("finished", [False, True])
def test_script_sanity(tmp_path, finished):
    run = _lineage_run(tmp_path)
    if finished:
        run.finish()
    script = _script(_render(run))
    for needle in ("sessionStorage", "location.reload", "replaceState", "hashchange"):
        assert needle in script, needle
    assert "innerHTML" not in script
    assert "http-equiv" not in script


def test_broken_json_files(tmp_path):
    run = SynRun(tmp_path)
    run.skill(0, "a").skill(0, "b").skill(1, "c", origin=_origin(["a"]))
    run.raw("rounds/000/skills/a.json", "{broken")
    run.raw("usage/target.json", "{broken")
    run.raw("rounds/000/usage.json", "not json at all")
    run.raw("rounds/001/results/gsm8k/c.jsonl", "{broken\n")
    out = write_report(run.path)
    assert Path(out).is_file()


def test_broken_run_json(tmp_path):
    run = SynRun(tmp_path)
    run.skill(0, "a")
    run.raw("run.json", "{broken")
    assert Path(write_report(run.path)).is_file()


# --- AC21: main / python -m ----------------------------------------------------------------------

def test_main_returns_path_and_prints(tmp_path, capsys):
    run = SynRun(tmp_path)
    run.skill(0, "a")
    out = main([str(run.path)])
    assert isinstance(out, Path)
    assert out.resolve() == (run.path / "report.html").resolve()
    assert out.is_file()
    assert str(out) in capsys.readouterr().out


def test_main_not_a_directory(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main([str(tmp_path / "missing")])
    assert exc.value.code == 2


def _subprocess_env():
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def test_python_m_report(tmp_path):
    run = SynRun(tmp_path)
    run.skill(0, "a").finish()
    proc = subprocess.run(
        [sys.executable, "-m", "skillopt.report", str(run.path)],
        cwd=str(PROJECT_ROOT), env=_subprocess_env(), capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=120,
    )
    assert proc.returncode == 0, (proc.stdout[-2000:], proc.stderr[-2000:])
    assert (run.path / "report.html").is_file()


# --- AC21: standalone module ------------------------------------------------------------------------

def test_report_source_imports_no_skillopt():
    src = (PROJECT_ROOT / "skillopt" / "report.py").read_text(encoding="utf-8")
    for pat in (r"^\s*from\s+skillopt\b", r"^\s*from\s+\.", r"^\s*import\s+skillopt\."):
        assert not re.search(pat, src, re.MULTILINE), pat


def test_report_import_loads_no_other_modules():
    code = ("import sys, skillopt.report; "
            "print(sorted(m for m in sys.modules if m.startswith('skillopt.') and m != 'skillopt.report'))")
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(PROJECT_ROOT), env=_subprocess_env(),
                          capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
    assert proc.returncode == 0, proc.stderr[-2000:]
    loaded = proc.stdout.strip()
    assert "skillopt.evaluator" not in loaded
    assert "skillopt.llm" not in loaded
    assert loaded == "[]"


# --- AC21: logging ------------------------------------------------------------------------------------

def _report_records(caplog):
    return [r for r in caplog.records if r.name == "skillopt.report" and r.getMessage().startswith("report: ")]


def test_logging_finished_info(tmp_path, caplog):
    run = SynRun(tmp_path)
    run.skill(0, "a").finish()
    with caplog.at_level(logging.DEBUG, logger="skillopt.report"):
        out = write_report(run.path)
    recs = _report_records(caplog)
    assert len(recs) == 1
    assert recs[0].levelno == logging.INFO
    assert recs[0].getMessage() == f"report: {out}"


def test_logging_running_debug(tmp_path, caplog):
    run = SynRun(tmp_path)
    run.skill(0, "a")
    with caplog.at_level(logging.DEBUG, logger="skillopt.report"):
        out = write_report(run.path)
    recs = _report_records(caplog)
    assert len(recs) == 1
    assert recs[0].levelno == logging.DEBUG
    assert recs[0].getMessage() == f"report: {out}"

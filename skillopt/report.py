"""Run report: one self-contained ``report.html`` built from the files already saved in a run directory.

Read-only and file based: it imports no other skillopt module and needs no model. The framework fixes only the
display format; it never judges, ranks or picks skills, and colors encode only ``origin.operation``. What a skill
was derived from is shown as the optimizer supplied it (``origin`` in ``skills/<id>.json``), falling back to
``parent_id``.

    python -m skillopt.report runs/<run>

Page layout (Python renders all markup; a small inline script only adds interaction):

- header: run name, optimizer, status;
- prompt-evolution tree (first and largest): one column per round, one node per evaluated (round, skill) with
  per-benchmark accuracy bars and avg output tokens, edges from parents (primary solid, others dashed). Vertical
  order is the saved order in the first round and the parent position in later rounds, never a score. Hover
  highlights ancestors and descendants; click opens the side detail panel (instruction, origin, word diff vs the
  first parent, metrics, up to ``MAX_FAILURES_SHOWN`` incorrect items per benchmark). Pan with drag, zoom with
  the wheel or the buttons;
- summary cards: round, status, skill counts, cumulative token usage;
- per-benchmark charts by round: accuracy and avg output tokens (clicking a point selects its node);
- a collapsed table of every evaluated skill in saved order; it is sortable only by the user (header clicks).

While a run is in progress (no ``run.json`` yet) the page reloads itself every ``REFRESH_SECONDS`` seconds and
keeps the selected node, the tree view, scroll positions and the table state (``sessionStorage`` + URL hash).
"""
from __future__ import annotations

import argparse
import difflib
import html
import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

REPORT_FILENAME: str = "report.html"
REFRESH_SECONDS: int = 5
MAX_FAILURES_SHOWN: int = 3
MAX_TEXT_CHARS: int = 300


# ----------------------------------------------------------------------------------------------------------- loading


def _read_json(path: Path) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _read_yaml(path: Path) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except (OSError, yaml.YAMLError):
        return None


def _as_dict(obj: Any) -> dict | None:
    return obj if isinstance(obj, dict) else None


def _as_int(obj: Any) -> int | None:
    return obj if isinstance(obj, int) and not isinstance(obj, bool) else None


def _read_failures(path: Path) -> tuple[list[dict], int]:
    """First ``MAX_FAILURES_SHOWN`` incorrect items (saved order) and the total number of incorrect items."""
    shown: list[dict] = []
    total = 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(item, dict) or item.get("correct"):
                    continue
                total += 1
                if len(shown) < MAX_FAILURES_SHOWN:
                    shown.append(item)
    except OSError:
        return [], 0
    return shown, total


def _origin_fields(skill_json: dict) -> tuple[str | None, str, list[str]]:
    """(operation, note, parent ids) from a saved skill json; parents fall back to ``parent_id``."""
    origin = skill_json.get("origin")
    if isinstance(origin, dict):
        op = origin.get("operation")
        note = origin.get("note")
        raw = origin.get("parents")
        parents = [p for p in raw if isinstance(p, str)] if isinstance(raw, (list, tuple)) else []
        return (op if isinstance(op, str) else None), (note if isinstance(note, str) else ""), parents
    pid = skill_json.get("parent_id")
    return None, "", ([pid] if isinstance(pid, str) and pid else [])


def _load_skill(rd: Path, sid: str, entry: dict) -> dict:
    sj = _as_dict(_read_json(rd / "skills" / f"{sid}.json")) or {}
    few_shot: list[dict] = []
    for fs in sj.get("few_shot") or []:
        if isinstance(fs, dict):
            few_shot.append({"question": str(fs.get("question", "")), "answer": str(fs.get("answer", ""))})
    operation, note, parents = _origin_fields(sj)
    results = _as_dict(entry.get("benchmarks")) or {}
    failures: dict[str, list[dict]] = {}
    num_failures: dict[str, int] = {}
    for bench in results:
        shown, total = _read_failures(rd / "results" / str(bench) / f"{sid}.jsonl")
        failures[bench] = shown
        num_failures[bench] = total
    instruction = sj.get("instruction")
    return {
        "id": sid,
        "instruction": instruction if isinstance(instruction, str) else "",
        "few_shot": few_shot,
        "num_tokens": _as_int(sj.get("num_tokens")),
        "metadata": _as_dict(sj.get("metadata")) or {},
        "operation": operation,
        "note": note,
        "parents": [{"id": p, "round": None} for p in parents],
        "children": [],
        "results": results,
        "failures": failures,
        "num_failures": num_failures,
    }


def load_run(run_dir: str | Path) -> dict:
    """Read a run directory into the fixed report data structure (see module docstring); never sorts skills."""
    path = Path(run_dir)
    if not path.is_dir():
        raise FileNotFoundError(f"run directory not found: {run_dir}")
    path = path.resolve()
    config = _as_dict(_read_yaml(path / "config.yaml")) or {}
    run_json = _as_dict(_read_json(path / "run.json"))
    opt_cfg = config.get("optimizer")
    opt_name = opt_cfg.get("name") if isinstance(opt_cfg, dict) else None
    if run_json is not None and _as_int(run_json.get("max_iterations")) is not None:
        max_iterations = _as_int(run_json.get("max_iterations"))
    else:
        max_iterations = _as_int(config.get("max_iterations"))
    stop_reason = run_json.get("stop_reason") if run_json is not None else None

    rounds: list[dict] = []
    benchmarks: list[str] = []
    rounds_dir = path / "rounds"
    if rounds_dir.is_dir():
        indexed = sorted(
            (int(d.name), d) for d in rounds_dir.iterdir() if d.is_dir() and d.name.isdigit()
        )
        for index, rd in indexed:
            summary = _read_json(rd / "summary.json")
            if not isinstance(summary, list):
                continue
            skills: list[dict] = []
            for entry in summary:
                if not isinstance(entry, dict) or not isinstance(entry.get("skill_id"), str):
                    continue
                skill = _load_skill(rd, entry["skill_id"], entry)
                for bench in skill["results"]:
                    if bench not in benchmarks:
                        benchmarks.append(bench)
                skills.append(skill)
            rounds.append({"index": index, "usage": _as_dict(_read_json(rd / "usage.json")), "skills": skills})

    # resolve parents to the most recent earlier round that evaluated them; then fill children
    seen_in: dict[str, list[int]] = {}
    by_ref: dict[tuple[str, int], dict] = {}
    for rnd in rounds:
        for skill in rnd["skills"]:
            for parent in skill["parents"]:
                earlier = seen_in.get(parent["id"], [])
                parent["round"] = earlier[-1] if earlier else None
        for skill in rnd["skills"]:
            rounds_of = seen_in.setdefault(skill["id"], [])
            if not rounds_of or rounds_of[-1] != rnd["index"]:
                rounds_of.append(rnd["index"])
            by_ref.setdefault((skill["id"], rnd["index"]), skill)
    for rnd in rounds:
        for skill in rnd["skills"]:
            for parent in skill["parents"]:
                if parent["round"] is None:
                    continue
                target = by_ref.get((parent["id"], parent["round"]))
                if target is None:
                    continue
                child = {"id": skill["id"], "round": rnd["index"]}
                if child not in target["children"]:
                    target["children"].append(child)

    usage_dir = path / "usage"
    return {
        "run_dir": str(path),
        "name": path.name,
        "optimizer": opt_name if isinstance(opt_name, str) else None,
        "max_iterations": max_iterations,
        "finished": run_json is not None,
        "stop_reason": stop_reason if isinstance(stop_reason, str) else None,
        "usage": {
            "target": _as_dict(_read_json(usage_dir / "target.json")),
            "optimizer": _as_dict(_read_json(usage_dir / "optimizer.json")),
        },
        "benchmarks": benchmarks,
        "rounds": rounds,
    }


# --------------------------------------------------------------------------------------------------------- rendering

_CSS = r"""
:root {
  color-scheme: light dark;
  --bg: #f6f7f9; --panel: #ffffff; --fg: #1f2328; --muted: #646c76; --border: #d9dee4; --soft: #eef1f4;
  --accent: #3b6ea8; --bar: #7b8794; --track: #e4e8ed; --edge: #a3acb6; --edge-hl: #3b6ea8; --grid: #e6e9ed;
  --ins: #d9e8fb; --del: #f6e3c6;
  --c0: #4e79a7; --c1: #e08a3c; --c2: #5fa8a2; --c3: #a57298; --c4: #c9a93a;
  --c5: #8f6f5a; --c6: #d78aa0; --c7: #6b8e4e; --none: #9aa1a9;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #15181c; --panel: #1d2126; --fg: #e3e6ea; --muted: #9aa3ad; --border: #343a42; --soft: #252a30;
    --accent: #7fa8d8; --bar: #8d99a6; --track: #2e343b; --edge: #5b6570; --edge-hl: #7fa8d8; --grid: #2a3037;
    --ins: #1f3a5c; --del: #4d3b1d;
    --c0: #7aa0cc; --c1: #e9a465; --c2: #7fc2bc; --c3: #c495b8; --c4: #dcc066;
    --c5: #b39380; --c6: #e6a7b9; --c7: #93b277; --none: #7d858e;
  }
}
.op-c0 { --op: var(--c0); } .op-c1 { --op: var(--c1); } .op-c2 { --op: var(--c2); } .op-c3 { --op: var(--c3); }
.op-c4 { --op: var(--c4); } .op-c5 { --op: var(--c5); } .op-c6 { --op: var(--c6); } .op-c7 { --op: var(--c7); }
.op-none { --op: var(--none); }
[hidden] { display: none !important; }
* { box-sizing: border-box; }
body {
  margin: 0; padding: 20px 28px 40px; background: var(--bg); color: var(--fg);
  font: 14.5px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
table, svg, .card-value, .num, dd { font-variant-numeric: tabular-nums; }
a { color: var(--accent); text-decoration: none; } a:hover { text-decoration: underline; }
h1 { font-size: 1.45em; margin: 0 0 2px; font-weight: 650; word-break: break-all; }
h2 { font-size: 1.1em; margin: 0; font-weight: 600; }
h3 { font-size: 1em; margin: 0 0 8px; font-weight: 600; }
section { margin-top: 22px; }
#header .meta { color: var(--muted); }
.section-head { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 18px; margin-bottom: 10px; }
.hint { color: var(--muted); font-size: 0.88em; }
#legend { list-style: none; display: flex; flex-wrap: wrap; gap: 4px 14px; margin: 0; padding: 0; font-size: 0.9em; }
#legend li { display: flex; align-items: center; gap: 6px; }
#legend .swatch { width: 11px; height: 11px; border-radius: 3px; background: var(--op); display: inline-block; }
.evo-body { display: grid; grid-template-columns: minmax(0, 1fr) 240px; gap: 16px; align-items: start; }
#evolution.open .evo-body { grid-template-columns: minmax(0, 1fr) 420px; }
@media (max-width: 900px) { .evo-body, #evolution.open .evo-body { grid-template-columns: minmax(0, 1fr); } }
.tree-wrap {
  position: relative; min-height: 280px; background: var(--panel);
  border: 1px solid var(--border); border-radius: 10px; overflow: hidden;
}
#tree { display: block; width: 100%; height: 100%; cursor: grab; touch-action: none; user-select: none; }
#tree.dragging { cursor: grabbing; }
.zoom { position: absolute; top: 8px; right: 8px; display: flex; gap: 4px; }
button {
  font: inherit; font-size: 0.85em; color: var(--fg); background: var(--panel); cursor: pointer;
  border: 1px solid var(--border); border-radius: 6px; padding: 2px 9px;
}
button:hover { background: var(--soft); }
.tree-wrap .empty { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
  margin: 0; color: var(--muted); pointer-events: none; }
.round-label { fill: var(--muted); font-size: 12px; font-weight: 600; }
.edge { fill: none; stroke: var(--edge); stroke-width: 1.6; }
.edge.secondary { stroke-width: 1; stroke-dasharray: 4 3; }
.edge.hl { stroke: var(--edge-hl); stroke-width: 2; }
.edge.secondary.hl { stroke-width: 1.4; }
.node { cursor: pointer; outline: none; }
.node .box { fill: var(--panel); stroke: var(--border); stroke-width: 1; }
.node .stripe { fill: var(--op); }
.node:hover .box, .node:focus .box { stroke: var(--muted); }
.node.selected .box { stroke: var(--accent); stroke-width: 2.2; }
.node text { fill: var(--fg); }
.node .node-id { font-size: 12px; font-weight: 600; }
.node .bench-label, .node .acc, .node .tok { font-size: 10px; fill: var(--muted); }
.bar .track { fill: var(--track); } .bar .fill { fill: var(--bar); }
#tree.dim .node:not(.hl) { opacity: 0.28; }
#tree.dim .edge:not(.hl) { opacity: 0.12; }
#detail {
  background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
  padding: 12px 14px; max-height: 80vh; overflow: auto;
}
.detail-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;
  color: var(--muted); font-size: 0.85em; }
#detail-empty { color: var(--muted); font-size: 0.92em; }
.skill h3 { word-break: break-all; }
.skill .meta { color: var(--muted); font-size: 0.88em; margin-bottom: 6px; }
.skill .lbl { color: var(--muted); margin-right: 4px; }
.skill > div { margin: 6px 0; }
.op { display: inline-block; font-size: 0.78em; font-weight: 500; padding: 0 7px; border-radius: 9px;
  border: 1px solid var(--op, var(--border)); color: var(--fg); margin-left: 6px; }
.size { color: var(--muted); font-weight: 400; font-size: 0.85em; margin-left: 6px; }
.unresolved { color: var(--muted); }
.note { color: var(--muted); font-style: italic; }
pre {
  white-space: pre-wrap; word-break: break-word; margin: 6px 0; padding: 8px 10px; font-size: 0.86em;
  background: var(--soft); border: 1px solid var(--border); border-radius: 6px;
  font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", monospace;
}
details > summary { cursor: pointer; color: var(--muted); }
.diff { font-size: 0.88em; line-height: 1.6; padding: 8px 10px; border: 1px dashed var(--border); border-radius: 6px; }
ins { background: var(--ins); text-decoration: none; border-radius: 3px; padding: 0 2px; }
del { background: var(--del); border-radius: 3px; padding: 0 2px; }
table { border-collapse: collapse; font-size: 0.86em; }
th, td { padding: 4px 8px; text-align: right; border-bottom: 1px solid var(--border); white-space: nowrap; }
th { font-weight: 600; color: var(--muted); }
th.l, td.l { text-align: left; }
.metrics { width: 100%; margin: 6px 0; }
.failures { margin: 6px 0; font-size: 0.9em; }
.more { color: var(--muted); font-size: 0.9em; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; }
.card { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; }
.card-label { color: var(--muted); font-size: 0.82em; text-transform: uppercase; letter-spacing: 0.04em; }
.card-value { font-size: 1.45em; font-weight: 600; margin: 2px 0; word-break: break-word; }
.card-sub { color: var(--muted); font-size: 0.85em; }
.card dl { display: grid; grid-template-columns: auto 1fr; gap: 0 10px; margin: 6px 0 0; font-size: 0.85em; }
.card dt { color: var(--muted); } .card dd { margin: 0; text-align: right; }
.bench-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); gap: 12px; margin-top: 10px; }
.bench { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; }
.charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 8px; }
.chart { width: 100%; height: auto; }
.chart text { fill: var(--muted); font-size: 10px; }
.chart .chart-title { fill: var(--fg); font-size: 11px; font-weight: 600; }
.chart .grid { stroke: var(--grid); stroke-width: 1; }
.chart .axis { stroke: var(--border); stroke-width: 1; }
.pt { fill: var(--op); fill-opacity: 0.85; stroke: var(--panel); stroke-width: 1; cursor: pointer; }
.pt.hl { stroke: var(--fg); stroke-width: 1.5; fill-opacity: 1; }
.pt.selected { stroke: var(--accent); stroke-width: 2; fill-opacity: 1; }
.empty-note { color: var(--muted); }
#table-details { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 10px 14px; }
#table-details > summary { font-weight: 600; color: var(--fg); }
.table-scroll { overflow: auto; margin-top: 10px; }
#skill-table th[data-sort-col] { cursor: pointer; user-select: none; }
#skill-table th[data-sort-col]::after { content: "\2195"; opacity: 0.3; margin-left: 4px; }
#skill-table th[data-dir="asc"]::after { content: "\25B2"; opacity: 0.9; }
#skill-table th[data-dir="desc"]::after { content: "\25BC"; opacity: 0.9; }
#skill-table tbody tr:hover { background: var(--soft); }
#skill-table th.group { text-align: center; color: var(--fg); }
"""

_JS = r"""
(function () {
  "use strict";
  const body = document.body;
  const storeKey = "skillopt-report:" + (body.dataset.run || "");
  const all = (sel, root) => Array.from((root || document).querySelectorAll(sel));

  function readState() {
    try {
      const raw = sessionStorage.getItem(storeKey);
      const parsed = raw ? JSON.parse(raw) : null;
      return parsed && typeof parsed === "object" ? parsed : {};
    } catch (e) { return {}; }
  }
  function saveState() {
    try { sessionStorage.setItem(storeKey, JSON.stringify(state)); } catch (e) { /* storage blocked */ }
  }
  const state = Object.assign(
    { sel: null, view: null, scrollY: 0, panelScroll: 0, tableOpen: false, sort: null }, readState());
  try { history.scrollRestoration = "manual"; } catch (e) { /* not supported */ }

  const svg = document.getElementById("tree");
  const viewport = document.getElementById("viewport");
  const evolution = document.getElementById("evolution");
  const detail = document.getElementById("detail");
  const detailEmpty = document.getElementById("detail-empty");
  const nodes = all("g.node", svg);
  const edges = all("path.edge", svg);
  const articles = all("article.skill", detail);
  const points = all("circle.pt");
  const nodeByKey = new Map(nodes.map((n) => [n.dataset.node, n]));
  const ups = new Map();
  const downs = new Map();
  edges.forEach((e) => {
    const src = e.dataset.src, dst = e.dataset.dst;
    if (!ups.has(dst)) ups.set(dst, []);
    if (!downs.has(src)) downs.set(src, []);
    ups.get(dst).push(src);
    downs.get(src).push(dst);
  });

  // ---- hover highlight: the node, all ancestors, all descendants and the edges between them
  function reach(start, graph) {
    const seen = new Set();
    const stack = [start];
    while (stack.length) {
      (graph.get(stack.pop()) || []).forEach((k) => { if (!seen.has(k)) { seen.add(k); stack.push(k); } });
    }
    return seen;
  }
  function highlight(key) {
    const lit = new Set([key, ...reach(key, ups), ...reach(key, downs)]);
    svg.classList.add("dim");
    nodes.forEach((n) => n.classList.toggle("hl", lit.has(n.dataset.node)));
    edges.forEach((e) => e.classList.toggle("hl", lit.has(e.dataset.src) && lit.has(e.dataset.dst)));
  }
  function clearHighlight() {
    svg.classList.remove("dim");
    nodes.forEach((n) => n.classList.remove("hl"));
    edges.forEach((e) => e.classList.remove("hl"));
  }

  // ---- pan / zoom
  const contentW = parseFloat(svg.dataset.width) || 0;
  const contentH = parseFloat(svg.dataset.height) || 0;
  const nodeW = parseFloat(svg.dataset.nodeW) || 0;
  let view = { x: 0, y: 0, k: 1 };
  let drag = null;
  const clampK = (k) => Math.min(3, Math.max(0.2, k));
  function applyView() {
    viewport.setAttribute("transform", "translate(" + view.x + "," + view.y + ") scale(" + view.k + ")");
    state.view = { x: view.x, y: view.y, k: view.k };
    saveState();
  }
  function box() { const r = svg.getBoundingClientRect(); return { w: r.width, h: r.height }; }
  function fit() {
    const s = box(), pad = 24;
    if (!contentW || !contentH || !s.w) return;
    const k = clampK(Math.min((s.w - 2 * pad) / contentW, (s.h - 2 * pad) / contentH, 1));
    view = { x: (s.w - contentW * k) / 2, y: Math.max(8, (s.h - contentH * k) / 2), k: k };
    applyView();
  }
  function defaultView() {
    const s = box();
    if (contentW > s.w - 24) {
      view = { x: s.w - contentW - 12, y: 0, k: 1 };
      applyView();
    } else {
      fit();
    }
  }
  function zoomAt(px, py, factor) {
    const k = clampK(view.k * factor);
    view = { x: px - (px - view.x) * k / view.k, y: py - (py - view.y) * k / view.k, k: k };
    applyView();
  }
  function reveal(key) {
    const n = nodeByKey.get(key);
    if (!n) return;
    const x = parseFloat(n.dataset.x), y = parseFloat(n.dataset.y), h = parseFloat(n.dataset.h);
    const s = box(), k = view.k;
    const sx = view.x + x * k, sy = view.y + y * k;
    if (sx < 0 || sy < 0 || sx + nodeW * k > s.w || sy + h * k > s.h) {
      view = { x: s.w / 2 - (x + nodeW / 2) * k, y: s.h / 2 - (y + h / 2) * k, k: k };
      applyView();
    }
  }
  svg.addEventListener("pointerdown", (e) => {
    if (e.button !== 0 || e.target.closest("g.node")) return;
    drag = { id: e.pointerId, x: e.clientX, y: e.clientY, vx: view.x, vy: view.y };
    svg.setPointerCapture(e.pointerId);
    svg.classList.add("dragging");
  });
  svg.addEventListener("pointermove", (e) => {
    if (!drag || e.pointerId !== drag.id) return;
    view = { x: drag.vx + e.clientX - drag.x, y: drag.vy + e.clientY - drag.y, k: view.k };
    applyView();
  });
  const endDrag = (e) => {
    if (drag && e.pointerId === drag.id) { drag = null; svg.classList.remove("dragging"); }
  };
  svg.addEventListener("pointerup", endDrag);
  svg.addEventListener("pointercancel", endDrag);
  svg.addEventListener("wheel", (e) => {
    e.preventDefault();
    const r = svg.getBoundingClientRect();
    zoomAt(e.clientX - r.left, e.clientY - r.top, Math.exp(-e.deltaY * 0.0015));
  }, { passive: false });
  all("button[data-zoom]").forEach((b) => b.addEventListener("click", () => {
    const s = box();
    if (b.dataset.zoom === "fit") fit();
    else zoomAt(s.w / 2, s.h / 2, b.dataset.zoom === "in" ? 1.25 : 0.8);
  }));

  // ---- selection and the detail panel
  function articleFor(key) {
    const el = key ? document.getElementById(key) : null;
    return el && el.matches("article.skill") ? el : null;
  }
  function select(key, opts) {
    opts = opts || {};
    const art = articleFor(key);
    if (!art) return false;
    articles.forEach((a) => { a.hidden = a !== art; });
    detailEmpty.hidden = true;
    evolution.classList.add("open");
    nodes.forEach((n) => n.classList.toggle("selected", n.dataset.node === key));
    points.forEach((p) => p.classList.toggle("selected", p.dataset.node === key));
    state.sel = key;
    if (!opts.keepHash) { try { history.replaceState(null, "", "#" + key); } catch (e) { /* ignore */ } }
    if (!opts.restoring) detail.scrollTop = 0;
    if (opts.reveal) reveal(key);
    if (opts.scroll) evolution.scrollIntoView({ block: "start" });
    saveState();
    return true;
  }
  function clearSelection() {
    articles.forEach((a) => { a.hidden = true; });
    detailEmpty.hidden = false;
    evolution.classList.remove("open");
    nodes.forEach((n) => n.classList.remove("selected"));
    points.forEach((p) => p.classList.remove("selected"));
    state.sel = null;
    try { history.replaceState(null, "", location.pathname + location.search); } catch (e) { /* ignore */ }
    saveState();
  }
  function hashKey() {
    const raw = location.hash.slice(1);
    try { return decodeURIComponent(raw); } catch (e) { return raw; }
  }
  nodes.forEach((n) => {
    const key = n.dataset.node;
    n.addEventListener("mouseenter", () => highlight(key));
    n.addEventListener("mouseleave", clearHighlight);
    n.addEventListener("focus", () => highlight(key));
    n.addEventListener("blur", clearHighlight);
    n.addEventListener("click", () => select(key));
    n.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); select(key); }
    });
  });
  points.forEach((p) => {
    const key = p.dataset.node;
    p.addEventListener("mouseenter", () => points.forEach((q) => q.classList.toggle("hl", q.dataset.node === key)));
    p.addEventListener("mouseleave", () => points.forEach((q) => q.classList.remove("hl")));
    p.addEventListener("click", () => select(key, { reveal: true, scroll: true }));
  });
  document.addEventListener("click", (e) => {
    const a = e.target.closest ? e.target.closest("a[href^='#']") : null;
    if (!a) return;
    const key = a.getAttribute("href").slice(1);
    if (!articleFor(key)) return;
    e.preventDefault();
    select(key, { reveal: true, scroll: !detail.contains(a) });
  });
  window.addEventListener("hashchange", () => { select(hashKey(), { reveal: true, keepHash: true }); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") clearSelection(); });
  document.getElementById("detail-close").addEventListener("click", clearSelection);

  // ---- table: user-only sorting (ascending -> descending -> saved order)
  const tableDetails = document.getElementById("table-details");
  const table = document.getElementById("skill-table");
  const tbody = table.tBodies[0];
  const heads = all("th[data-sort-col]", table);
  function cellValue(row, col, type) {
    const td = row.cells[col];
    const raw = td ? td.dataset.value : "";
    if (raw === undefined || raw === "") return null;
    if (type === "num") { const v = parseFloat(raw); return isNaN(v) ? null : v; }
    return raw.toLowerCase();
  }
  function applySort() {
    const s = state.sort;
    heads.forEach((h) => {
      if (s && h.dataset.sortCol === String(s.col)) h.dataset.dir = s.dir;
      else h.removeAttribute("data-dir");
    });
    const order = (a, b) => Number(a.dataset.order) - Number(b.dataset.order);
    let cmp = order;
    if (s) {
      const head = heads.find((h) => h.dataset.sortCol === String(s.col));
      const type = head ? head.dataset.sortType : "text";
      const sign = s.dir === "desc" ? -1 : 1;
      cmp = (a, b) => {
        const va = cellValue(a, s.col, type), vb = cellValue(b, s.col, type);
        if (va === null || vb === null) return va === vb ? order(a, b) : (va === null ? 1 : -1);
        if (va < vb) return -sign;
        if (va > vb) return sign;
        return order(a, b);
      };
    }
    Array.from(tbody.rows).sort(cmp).forEach((r) => tbody.appendChild(r));
  }
  heads.forEach((h) => h.addEventListener("click", () => {
    const col = parseInt(h.dataset.sortCol, 10);
    const s = state.sort;
    if (!s || s.col !== col) state.sort = { col: col, dir: "asc" };
    else if (s.dir === "asc") state.sort = { col: col, dir: "desc" };
    else state.sort = null;
    applySort();
    saveState();
  }));
  tableDetails.addEventListener("toggle", () => { state.tableOpen = tableDetails.open; saveState(); });

  // ---- restore state (hash wins over the stored selection), then keep saving it
  const savedY = Number(state.scrollY) || 0;
  const savedPanel = Number(state.panelScroll) || 0;
  if (state.tableOpen) tableDetails.open = true;
  if (state.sort && typeof state.sort.col === "number") applySort(); else state.sort = null;
  const fromHash = hashKey();
  if (!(fromHash && select(fromHash, { keepHash: true, restoring: true }))) {
    if (!(state.sel && select(state.sel, { restoring: true }))) state.sel = null;
  }
  const v = state.view;
  if (v && isFinite(v.x) && isFinite(v.y) && isFinite(v.k)) {
    view = { x: Number(v.x), y: Number(v.y), k: clampK(Number(v.k)) };
    applyView();
  } else {
    defaultView();
  }
  requestAnimationFrame(() => {
    window.scrollTo(0, savedY);
    detail.scrollTop = savedPanel;
    state.scrollY = savedY;
    state.panelScroll = savedPanel;
    saveState();
    let pending = false;
    window.addEventListener("scroll", () => {
      if (pending) return;
      pending = true;
      requestAnimationFrame(() => { pending = false; state.scrollY = window.scrollY; saveState(); });
    });
    detail.addEventListener("scroll", () => { state.panelScroll = detail.scrollTop; saveState(); });
  });

  // ---- auto reload while the run is in progress
  const seconds = parseInt(body.dataset.refreshSeconds || "", 10);
  if (seconds > 0) {
    const reload = () => {
      if (drag) { setTimeout(reload, 1000); return; }
      state.scrollY = window.scrollY;
      state.panelScroll = detail.scrollTop;
      saveState();
      location.reload();
    };
    setTimeout(reload, seconds * 1000);
  }
})();
"""

# tree geometry (px, in tree coordinates)
_NODE_W = 240
_COL_GAP = 90
_ROW_GAP = 18
_PAD_X = 24
_PAD_TOP = 40
_ROW_TOP = 26
_BENCH_ROW = 16
_NODE_PAD_B = 8
_TRACK_X, _TRACK_W = 78, 70

# chart geometry
_CW, _CH, _CML, _CMR, _CMT, _CMB = 340, 200, 48, 14, 28, 30


def _e(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _clip(value: Any) -> str:
    text = "" if value is None else str(value)
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS] + "..."
    return text


def _shorten(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _raw(value: Any) -> str:
    """A numeric value as saved (for ``data-value``), or ``""`` when missing / not numeric."""
    return "" if _num(value) is None else str(value)


def _pct(value: Any) -> str:
    v = _num(value)
    return "-" if v is None else f"{v * 100:.1f}%"


def _dec(value: Any) -> str:
    v = _num(value)
    return "-" if v is None else f"{v:,.1f}"


def _int_txt(value: Any) -> str:
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value:,}"
    v = _num(value)
    return "-" if v is None else f"{round(v):,}"


def _anchor(round_index: int, skill_id: str) -> str:
    return f"r{round_index:03d}-{skill_id}"


def _result(skill: dict, bench: str) -> dict:
    res = skill["results"].get(bench)
    return res if isinstance(res, dict) else {}


def _op_classes(data: dict) -> dict[str | None, str]:
    """Operation -> color class, in first-seen order (``None`` -> ``op-none``)."""
    classes: dict[str | None, str] = {}
    count = 0
    for rnd in data["rounds"]:
        for skill in rnd["skills"]:
            op = skill["operation"] or None
            if op in classes:
                continue
            if op is None:
                classes[None] = "op-none"
            else:
                classes[op] = f"op-c{count % 8}"
                count += 1
    return classes


def _node_h(n_bench: int) -> int:
    return _ROW_TOP + _BENCH_ROW * n_bench + _NODE_PAD_B


def _layout(data: dict) -> dict:
    """Node positions: one column per round; slots follow saved order (first round) or parent slots (later rounds)."""
    nh = _node_h(len(data["benchmarks"]))
    nodes: dict[str, dict] = {}
    columns: list[tuple[int, list[str]]] = []
    for col, rnd in enumerate(data["rounds"]):
        idx = rnd["index"]
        entries: list[tuple[str, dict]] = []
        seen: set[str] = set()
        for skill in rnd["skills"]:
            key = _anchor(idx, skill["id"])
            if key in seen or key in nodes:
                continue
            seen.add(key)
            entries.append((key, skill))
        if col > 0:

            def sort_key(entry: tuple[str, dict]) -> tuple:
                for parent in entry[1]["parents"]:
                    if parent["round"] is None:
                        continue
                    pinfo = nodes.get(_anchor(parent["round"], parent["id"]))
                    if pinfo is not None:
                        return (0, pinfo["slot"], parent["round"])
                return (1,)

            entries = sorted(entries, key=sort_key)
        keys = []
        for slot, (key, skill) in enumerate(entries):
            nodes[key] = {
                "round": idx,
                "skill": skill,
                "slot": slot,
                "x": _PAD_X + col * (_NODE_W + _COL_GAP),
                "y": _PAD_TOP + slot * (nh + _ROW_GAP),
            }
            keys.append(key)
        columns.append((idx, keys))
    ncols = len(columns)
    max_rows = max((len(keys) for _, keys in columns), default=0)
    width = 2 * _PAD_X + ncols * _NODE_W + max(ncols - 1, 0) * _COL_GAP if ncols else 0
    height = _PAD_TOP + max_rows * (nh + _ROW_GAP) - _ROW_GAP + 20 if max_rows else 0
    return {"nodes": nodes, "columns": columns, "width": width, "height": height, "node_h": nh}


def _render_header(data: dict) -> str:
    status = f"finished: {data['stop_reason']}" if data["finished"] else "running"
    return (
        f'<header id="header"><h1>{_e(data["name"])}</h1>'
        f'<div class="meta">optimizer: {_e(data["optimizer"] or "-")} &middot; {_e(status)}</div></header>'
    )


def _render_node(key: str, info: dict, benches: list[str], cls: str, nh: int) -> str:
    skill = info["skill"]
    sid = skill["id"]
    op = skill["operation"] or ""
    title = [sid, f"round {info['round']}, operation: {op or 'unknown'}"]
    bars: list[str] = []
    for i, bench in enumerate(benches):
        res = _result(skill, bench)
        acc = _num(res.get("accuracy"))
        tok = res.get("avg_output_tokens")
        title.append(f"{bench}: acc {_pct(acc)}, avg out {_int_txt(tok)} tokens")
        y0 = _ROW_TOP + i * _BENCH_ROW
        fill = 0.0 if acc is None else _TRACK_W * min(max(acc, 0.0), 1.0)
        bars.append(
            f'<g class="bar" data-benchmark="{_e(bench)}" data-value="{_e(_raw(res.get("accuracy")))}">'
            f'<text class="bench-label" x="14" y="{y0 + 10}">{_e(_shorten(bench, 10))}</text>'
            f'<rect class="track" x="{_TRACK_X}" y="{y0 + 3}" width="{_TRACK_W}" height="8" rx="2"/>'
            f'<rect class="fill" x="{_TRACK_X}" y="{y0 + 3}" width="{fill:.1f}" height="8" rx="2"/>'
            f'<text class="acc" x="{_TRACK_X + _TRACK_W + 6}" y="{y0 + 10}">{_pct(acc)}</text>'
            f'<text class="tok" x="{_NODE_W - 8}" y="{y0 + 10}" text-anchor="end">{_int_txt(tok)}</text></g>'
        )
    stripe = f"M6,0.5 A5.5,5.5 0 0 0 0.5,6 V{nh - 6} A5.5,5.5 0 0 0 6,{nh - 0.5} Z"
    return (
        f'<g class="node {cls}" data-node="{_e(key)}" data-round="{info["round"]}" data-skill="{_e(sid)}" '
        f'data-op="{_e(op)}" data-slot="{info["slot"]}" data-x="{info["x"]}" data-y="{info["y"]}" '
        f'data-h="{nh}" tabindex="0" transform="translate({info["x"]},{info["y"]})">'
        f"<title>{_e(chr(10).join(title))}</title>"
        f'<rect class="box" x="0.5" y="0.5" width="{_NODE_W - 1}" height="{nh - 1}" rx="6"/>'
        f'<path class="stripe" d="{stripe}"/>'
        f'<text class="node-id" x="14" y="17">{_e(_shorten(sid, 28))}</text>'
        + "".join(bars)
        + "</g>"
    )


def _render_edges(layout: dict) -> str:
    nodes = layout["nodes"]
    half = layout["node_h"] / 2
    out: list[str] = []
    for key, info in nodes.items():
        seen: set[str] = set()
        for i, parent in enumerate(info["skill"]["parents"]):
            if parent["id"] in seen:
                continue
            seen.add(parent["id"])
            if parent["round"] is None:
                continue
            src = _anchor(parent["round"], parent["id"])
            pinfo = nodes.get(src)
            if pinfo is None:
                continue
            x1, y1 = pinfo["x"] + _NODE_W, pinfo["y"] + half
            x2, y2 = info["x"], info["y"] + half
            mx = (x1 + x2) / 2
            kind = "primary" if i == 0 else "secondary"
            out.append(
                f'<path class="edge {kind}" data-src="{_e(src)}" data-dst="{_e(key)}" '
                f'd="M{x1:.1f},{y1:.1f} C{mx:.1f},{y1:.1f} {mx:.1f},{y2:.1f} {x2:.1f},{y2:.1f}"/>'
            )
    return "".join(out)


def _render_tree(data: dict, layout: dict, classes: dict) -> str:
    nh = layout["node_h"]
    benches = data["benchmarks"]
    labels = "".join(
        f'<text class="round-label" x="{_PAD_X + col * (_NODE_W + _COL_GAP) + 2}" y="24">Round {idx}</text>'
        for col, (idx, _) in enumerate(layout["columns"])
    )
    node_svg = "".join(
        _render_node(key, info, benches, classes.get(info["skill"]["operation"] or None, "op-none"), nh)
        for key, info in layout["nodes"].items()
    )
    legend = "".join(
        f'<li data-op="{_e(op or "")}" class="{cls}"><span class="swatch"></span>{_e(op or "unknown")}</li>'
        for op, cls in classes.items()
    )
    empty = "" if layout["nodes"] else '<p class="empty">No rounds evaluated yet.</p>'
    wrap_h = max(layout["height"] + 20, 280)
    return (
        '<section id="evolution"><div class="section-head"><h2>Prompt evolution</h2>'
        f'<ul id="legend">{legend}</ul>'
        '<span class="hint">Drag to pan, scroll to zoom, click a node for details.</span></div>'
        f'<div class="evo-body"><div class="tree-wrap" style="height: min(72vh, {wrap_h}px)">'
        f'<svg id="tree" data-width="{layout["width"]}" data-height="{layout["height"]}" data-node-w="{_NODE_W}" '
        'role="img" aria-label="Prompt evolution tree">'
        f'<g id="viewport"><g class="labels">{labels}</g><g class="edges">{_render_edges(layout)}</g>'
        f'<g class="nodes">{node_svg}</g></g></svg>'
        '<div class="zoom"><button type="button" data-zoom="in" title="Zoom in">+</button>'
        '<button type="button" data-zoom="out" title="Zoom out">&minus;</button>'
        '<button type="button" data-zoom="fit" title="Fit the whole tree">Fit</button></div>'
        f"{empty}</div>"
        + _render_detail(data, layout)
        + "</div></section>"
    )


def _parent_links(parents: list[dict]) -> str:
    parts = []
    for p in parents:
        if p["round"] is None:
            parts.append(f'<span class="unresolved">{_e(p["id"])}</span>')
        else:
            parts.append(f'<a href="#{_e(_anchor(p["round"], p["id"]))}">{_e(p["id"])}</a>')
    return ", ".join(parts) if parts else "-"


def _word_diff(old: str, new: str) -> str:
    a, b = old.split(), new.split()
    out: list[str] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag == "equal":
            out.append(_e(" ".join(a[i1:i2])))
            continue
        if tag in ("delete", "replace"):
            out.append(f"<del>{_e(' '.join(a[i1:i2]))}</del>")
        if tag in ("insert", "replace"):
            out.append(f"<ins>{_e(' '.join(b[j1:j2]))}</ins>")
    return " ".join(out)


def _render_failures(bench: str, items: list[dict], total: int) -> str:
    out = [
        f'<details class="failures" data-benchmark="{_e(bench)}"><summary>{_e(bench)}: {total} incorrect</summary>'
    ]
    for item in items:
        out.append(
            "<pre>"
            f"question: {_e(_clip(item.get('question')))}\n"
            f"response: {_e(_clip(item.get('response_text')))}\n"
            f"prediction: {_e(_clip(item.get('prediction')))}\n"
            f"gold: {_e(_clip(item.get('gold')))}\n"
            f"truncated: {_e(bool(item.get('truncated')))}"
            "</pre>"
        )
    if total > len(items):
        out.append(f'<div class="more">+{total - len(items)} more</div>')
    out.append("</details>")
    return "".join(out)


def _render_article(key: str, rnd_index: int, skill: dict, benchmarks: list[str], by_ref: dict) -> str:
    sid = skill["id"]
    size = skill["num_tokens"]
    size_html = f'<span class="size">{_int_txt(size)} tokens</span>' if size is not None else '<span class="size">-</span>'
    op_html = f'<span class="op">{_e(skill["operation"])}</span>' if skill["operation"] else ""
    out = [
        f'<article class="skill" id="{_e(key)}" data-node="{_e(key)}" hidden>',
        f"<h3>{_e(sid)}{op_html}{size_html}</h3>",
        f'<div class="meta">Round {rnd_index}</div>',
        f'<div class="parents"><span class="lbl">parents:</span> {_parent_links(skill["parents"])}</div>',
    ]
    if skill["children"]:
        links = ", ".join(
            f'<a href="#{_e(_anchor(c["round"], c["id"]))}">{_e(c["id"])}</a>' for c in skill["children"]
        )
        out.append(f'<div class="children"><span class="lbl">children:</span> {links}</div>')
    if skill["note"]:
        out.append(f'<div class="note">{_e(skill["note"])}</div>')
    out.append(f'<pre class="instruction">{_e(skill["instruction"])}</pre>')
    if skill["few_shot"]:
        demos = "".join(f"<pre>Q: {_e(fs['question'])}\nA: {_e(fs['answer'])}</pre>" for fs in skill["few_shot"])
        out.append(
            f'<details class="few-shot"><summary>few-shot demos ({len(skill["few_shot"])})</summary>{demos}</details>'
        )
    if skill["parents"] and skill["parents"][0]["round"] is not None:
        first = skill["parents"][0]
        parent = by_ref.get((first["id"], first["round"]))
        if parent is not None:
            out.append(
                f'<div class="diff"><span class="lbl">vs {_e(first["id"])}:</span> '
                f'{_word_diff(parent["instruction"], skill["instruction"])}'
                f'<br><span class="lbl">few-shot: {len(skill["few_shot"])} (parent: {len(parent["few_shot"])})</span>'
                "</div>"
            )
    rows = []
    for bench in benchmarks:
        res = skill["results"].get(bench)
        if not isinstance(res, dict):
            continue
        rows.append(
            f'<tr data-benchmark="{_e(bench)}"><td class="l">{_e(bench)}</td><td>{_int_txt(res.get("n"))}</td>'
            f'<td>{_pct(res.get("accuracy"))}</td><td>{_dec(res.get("avg_input_tokens"))}</td>'
            f'<td>{_dec(res.get("avg_output_tokens"))}</td><td>{_dec(res.get("avg_reasoning_tokens"))}</td>'
            f'<td>{_pct(res.get("truncation_rate"))}</td></tr>'
        )
    if rows:
        out.append(
            '<table class="metrics"><thead><tr><th class="l">benchmark</th><th>n</th><th>acc</th><th>in</th>'
            "<th>out</th><th>think</th><th>trunc</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
        )
    for bench in benchmarks:
        total = skill["num_failures"].get(bench, 0)
        if total > 0:
            out.append(_render_failures(bench, skill["failures"].get(bench, []), total))
    out.append("</article>")
    return "".join(out)


def _render_detail(data: dict, layout: dict) -> str:
    by_ref = {(s["id"], r["index"]): s for r in data["rounds"] for s in r["skills"]}
    articles = "".join(
        _render_article(key, info["round"], info["skill"], data["benchmarks"], by_ref)
        for key, info in layout["nodes"].items()
    )
    return (
        '<aside id="detail"><div class="detail-head"><span>Details</span>'
        '<button type="button" id="detail-close" title="Close (Esc)">Close</button></div>'
        '<div id="detail-empty">Click a node in the tree (or a chart point) to see its prompt, origin and results.</div>'
        f"{articles}</aside>"
    )


def _usage_card(card: str, label: str, usage: dict | None) -> str:
    if not usage:
        return (
            f'<div class="card" data-card="{card}"><div class="card-label">{label}</div>'
            '<div class="card-value">-</div><div class="card-sub">no usage recorded</div></div>'
        )
    lat = _num(usage.get("latency"))
    rows = [
        ("calls", _int_txt(usage.get("calls"))),
        ("input", _int_txt(usage.get("input_tokens"))),
        ("output", _int_txt(usage.get("output_tokens"))),
        ("reasoning", _int_txt(usage.get("reasoning_tokens"))),
        ("time", "-" if lat is None else f"{lat:,.1f}s"),
    ]
    dl = "".join(f"<dt>{k}</dt><dd>{_e(v)}</dd>" for k, v in rows)
    return (
        f'<div class="card" data-card="{card}"><div class="card-label">{label}</div>'
        f'<div class="card-value">{_e(_int_txt(usage.get("total_tokens")))}</div>'
        f'<div class="card-sub">total tokens</div><dl>{dl}</dl></div>'
    )


def _render_summary(data: dict) -> str:
    rounds = data["rounds"]
    latest = rounds[-1]["index"] if rounds else None
    max_it = data["max_iterations"]
    n_eval = sum(len(r["skills"]) for r in rounds)
    n_distinct = len({s["id"] for r in rounds for s in r["skills"]})
    status = f"finished: {data['stop_reason']}" if data["finished"] else "running"
    return (
        '<section id="summary"><div class="cards">'
        '<div class="card" data-card="round"><div class="card-label">Round</div>'
        f'<div class="card-value">{_e(_int_txt(latest))} / {_e(_int_txt(max_it))}</div>'
        '<div class="card-sub">latest / max iterations</div></div>'
        '<div class="card" data-card="status"><div class="card-label">Status</div>'
        f'<div class="card-value">{_e(status)}</div></div>'
        '<div class="card" data-card="skills"><div class="card-label">Skills</div>'
        f'<div class="card-value">{_int_txt(n_eval)}</div>'
        f'<div class="card-sub">evaluated (round, skill) pairs &middot; {_int_txt(n_distinct)} distinct skills</div></div>'
        + _usage_card("target-usage", "Target usage", data["usage"].get("target"))
        + _usage_card("optimizer-usage", "Optimizer usage", data["usage"].get("optimizer"))
        + "</div></section>"
    )


def _nice_max(v: float) -> float:
    if v <= 0:
        return 1.0
    mag = 10 ** math.floor(math.log10(v))
    for step in (1, 2, 2.5, 5, 10):
        if v <= step * mag:
            return step * mag
    return 10 * mag


def _render_chart(metric: str, title: str, bench: str, data: dict, layout: dict, classes: dict) -> str:
    indexes = [idx for idx, _ in layout["columns"]]
    rmin, rmax = (min(indexes), max(indexes)) if indexes else (0, 1)
    pw, ph = _CW - _CML - _CMR, _CH - _CMT - _CMB
    inner = 12.0
    span = (pw - 2 * inner) / (rmax - rmin) if rmax > rmin else pw - 2 * inner

    def sx(r: float) -> float:
        if rmax == rmin:
            return _CML + pw / 2
        return _CML + inner + (pw - 2 * inner) * (r - rmin) / (rmax - rmin)

    pts: list[tuple[float, float, str, dict, str]] = []
    for idx, keys in layout["columns"]:
        n = len(keys)
        step = min(5.0, span * 0.6 / max(n, 1))
        for key in keys:
            info = layout["nodes"][key]
            value = _num(_result(info["skill"], bench).get(metric))
            if value is None:
                continue
            x = sx(idx) + (info["slot"] - (n - 1) / 2) * step
            pts.append((x, value, key, info, classes.get(info["skill"]["operation"] or None, "op-none")))

    if metric == "accuracy":
        y_max = 100.0
        scale = 100.0
        ticks = [(0.0, "0%"), (50.0, "50%"), (100.0, "100%")]
    else:
        scale = 1.0
        y_max = _nice_max(max((v for _, v, _, _, _ in pts), default=0.0))
        ticks = [(0.0, "0"), (y_max / 2, _int_txt(y_max / 2)), (y_max, _int_txt(y_max))]

    def sy(y: float) -> float:
        return _CMT + ph * (1.0 - min(max(y / y_max, 0.0), 1.0))

    out = [
        f'<svg class="chart" data-metric="{metric}" viewBox="0 0 {_CW} {_CH}" width="{_CW}" height="{_CH}" '
        f'role="img" aria-label="{_e(bench)}: {_e(title)}">',
        f'<text class="chart-title" x="{_CML}" y="14">{_e(title)}</text>',
    ]
    for value, label in ticks:
        y = sy(value)
        out.append(
            f'<line class="grid" x1="{_CML}" y1="{y:.1f}" x2="{_CML + pw}" y2="{y:.1f}"/>'
            f'<text x="{_CML - 6}" y="{y + 3:.1f}" text-anchor="end">{_e(label)}</text>'
        )
    out.append(f'<line class="axis" x1="{_CML}" y1="{_CMT + ph}" x2="{_CML + pw}" y2="{_CMT + ph}"/>')
    every = max(1, math.ceil(len(indexes) / 8))
    for i, idx in enumerate(indexes):
        if i % every == 0 or i == len(indexes) - 1:
            out.append(f'<text x="{sx(idx):.1f}" y="{_CMT + ph + 14}" text-anchor="middle">{idx}</text>')
    out.append(f'<text x="{_CML + pw / 2:.1f}" y="{_CH - 3}" text-anchor="middle">round</text>')
    for x, value, key, info, cls in pts:
        sid = info["skill"]["id"]
        shown = _pct(value) if metric == "accuracy" else f"{_dec(value)} tokens"
        out.append(
            f'<circle class="pt {cls}" data-node="{_e(key)}" data-round="{info["round"]}" data-skill="{_e(sid)}" '
            f'cx="{x:.1f}" cy="{sy(value * scale):.1f}" r="4.5">'
            f"<title>{_e(sid)} (round {info['round']}): {_e(shown)}</title></circle>"
        )
    out.append("</svg>")
    return "".join(out)


def _render_trends(data: dict, layout: dict, classes: dict) -> str:
    out = ['<section id="trends"><h2>Benchmarks by round</h2>']
    if not data["benchmarks"]:
        out.append('<p class="empty-note">No benchmark results yet.</p>')
    out.append('<div class="bench-grid">')
    for bench in data["benchmarks"]:
        out.append(
            f'<div class="bench" data-benchmark="{_e(bench)}"><h3>{_e(bench)}</h3><div class="charts">'
            + _render_chart("accuracy", "accuracy", bench, data, layout, classes)
            + _render_chart("avg_output_tokens", "avg output tokens (thinking incl.)", bench, data, layout, classes)
            + "</div></div>"
        )
    out.append("</div></section>")
    return "".join(out)


def _num_td(value: Any, text: str) -> str:
    return f'<td data-value="{_e(_raw(value))}">{_e(text)}</td>'


def _render_table(data: dict) -> str:
    benches = data["benchmarks"]
    head1 = (
        '<tr><th class="l" rowspan="2" data-sort-col="0" data-sort-type="num">Round</th>'
        '<th class="l" rowspan="2" data-sort-col="1" data-sort-type="text">Skill</th>'
        '<th class="l" rowspan="2" data-sort-col="2" data-sort-type="text">Operation</th>'
        '<th rowspan="2" data-sort-col="3" data-sort-type="num">Size (tokens)</th>'
        + "".join(f'<th class="group" colspan="5">{_e(b)}</th>' for b in benches)
        + "</tr>"
    )
    sub = ("acc", "in", "out", "think", "trunc rate")
    head2 = "<tr>" + "".join(
        f'<th data-sort-col="{4 + 5 * b + j}" data-sort-type="num">{name}</th>'
        for b in range(len(benches))
        for j, name in enumerate(sub)
    ) + "</tr>"
    rows: list[str] = []
    order = 0
    for rnd in data["rounds"]:
        idx = rnd["index"]
        for skill in rnd["skills"]:
            sid = skill["id"]
            key = _anchor(idx, sid)
            op = skill["operation"] or ""
            cells = [
                f'<td class="l" data-value="{idx}">{idx}</td>',
                f'<td class="l" data-value="{_e(sid)}"><a href="#{_e(key)}">{_e(sid)}</a></td>',
                f'<td class="l" data-value="{_e(op)}">{_e(op or "-")}</td>',
                _num_td(skill["num_tokens"], _int_txt(skill["num_tokens"])),
            ]
            for bench in benches:
                res = _result(skill, bench)
                cells.append(
                    _num_td(res.get("accuracy"), _pct(res.get("accuracy")))
                    + _num_td(res.get("avg_input_tokens"), _dec(res.get("avg_input_tokens")))
                    + _num_td(res.get("avg_output_tokens"), _dec(res.get("avg_output_tokens")))
                    + _num_td(res.get("avg_reasoning_tokens"), _dec(res.get("avg_reasoning_tokens")))
                    + _num_td(res.get("truncation_rate"), _pct(res.get("truncation_rate")))
                )
            rows.append(
                f'<tr class="row" data-round="{idx}" data-skill="{_e(sid)}" data-node="{_e(key)}" '
                f'data-order="{order}">{"".join(cells)}</tr>'
            )
            order += 1
    return (
        '<section id="table"><details id="table-details">'
        f"<summary>All evaluated skills ({order}) &middot; click a column header to sort</summary>"
        f'<div class="table-scroll"><table id="skill-table"><thead>{head1}{head2}</thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div></details></section>'
    )


def render_report(data: dict) -> str:
    """Render the ``load_run`` structure as one self-contained HTML page (inline CSS, SVG and JS only)."""
    running = not data["finished"]
    refresh = f'<noscript><meta http-equiv="refresh" content="{REFRESH_SECONDS}"></noscript>' if running else ""
    body_attrs = f'data-run="{_e(data["name"])}"'
    if running:
        body_attrs += f' data-refresh-seconds="{REFRESH_SECONDS}"'
    layout = _layout(data)
    classes = _op_classes(data)
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"{refresh}<title>{_e(data['name'])} - run report</title><style>{_CSS}</style></head>"
        f"<body {body_attrs}>"
        + _render_header(data)
        + _render_tree(data, layout, classes)
        + _render_summary(data)
        + _render_trends(data, layout, classes)
        + _render_table(data)
        + f"<script>{_JS}</script></body></html>\n"
    )


def write_report(run_dir: str | Path) -> Path:
    """Write ``<run>/report.html`` atomically (via ``report.html.tmp`` + ``os.replace``); returns its path."""
    data = load_run(run_dir)
    root = Path(data["run_dir"])
    out = root / REPORT_FILENAME
    tmp = root / f"{REPORT_FILENAME}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(render_report(data))
        os.replace(tmp, out)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    logger.log(logging.INFO if data["finished"] else logging.DEBUG, f"report: {out}")
    return out


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(
        prog="python -m skillopt.report", description="Write <run>/report.html from a saved run directory."
    )
    parser.add_argument("run_dir", help="run directory, e.g. runs/<run>")
    args = parser.parse_args(argv)
    if not Path(args.run_dir).is_dir():
        parser.error(f"not a directory: {args.run_dir}")
    path = write_report(args.run_dir)
    print(path)
    return path


if __name__ == "__main__":
    main()

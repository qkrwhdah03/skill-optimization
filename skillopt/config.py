"""YAML config loading with model presets and dotted ``--set`` overrides."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def apply_overrides(config: dict, overrides: list[str]) -> dict:
    out = copy.deepcopy(config)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"invalid override '{item}': expected KEY=VALUE (e.g. optimize.stop.max_iterations=5)")
        key, raw = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"invalid override '{item}': empty key")
        value = yaml.safe_load(raw)
        parts = key.split(".")
        node = out
        for p in parts[:-1]:
            if isinstance(node, list):
                node = node[_list_index(node, p, key)]
                continue
            if not isinstance(node.get(p), (dict, list)):
                node[p] = {}
            node = node[p]
        last = parts[-1]
        if isinstance(node, list):
            node[_list_index(node, last, key)] = value
        else:
            node[last] = value
    return out


def _list_index(node: list, part: str, key: str) -> int:
    """Index for path component ``part`` on a list node; ``ValueError`` if not a valid in-range integer."""
    if not part.isdecimal():
        raise ValueError(f"invalid override key '{key}': '{part}' must be an integer index into a list")
    idx = int(part)
    if idx >= len(node):
        raise ValueError(f"invalid override key '{key}': index {idx} out of range (list has {len(node)} items)")
    return idx


def resolve_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _read_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def load_config(
    path: str | Path,
    overrides: list[str] | None = None,
    presets_dir: str | Path | None = None,
) -> dict:
    """Read YAML -> apply ``--set`` overrides -> resolve ``llm.<role>.preset`` (explicit role keys win)."""
    path = config_path(path)
    config = apply_overrides(_read_yaml(path), overrides or [])
    pdir = Path(presets_dir) if presets_dir is not None else path.parent / "models"
    llm = config.get("llm") or {}
    resolved: dict = {}
    for role, role_cfg in llm.items():
        role_cfg = dict(role_cfg or {})
        if "preset" in role_cfg:
            preset = role_cfg.pop("preset")
            preset_path = pdir / f"{preset}.yaml"
            if not preset_path.is_file():
                raise FileNotFoundError(f"model preset '{preset}' not found at {preset_path}")
            role_cfg = deep_merge(_read_yaml(preset_path), role_cfg)
        resolved[role] = role_cfg
    if "llm" in config:
        config["llm"] = resolved
    return config


def config_path(path: str | Path) -> Path:
    """Resolve a config path: as given if it exists or is absolute, else relative to the project root."""
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = resolve_path(path)
    return path


def read_raw_config(path: str | Path) -> dict:
    """The raw YAML of a config file (no overrides, no preset resolution)."""
    return _read_yaml(config_path(path))

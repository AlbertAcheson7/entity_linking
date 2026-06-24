from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    root = (config_path.parent.parent / cfg.get("project_root", ".")).resolve()
    cfg["_config_path"] = str(config_path)
    cfg["_root"] = str(root)
    for key, value in cfg["paths"].items():
        cfg["paths"][key] = str((root / value).resolve())
    os.environ.setdefault("HF_ENDPOINT", cfg["models"]["hf_endpoint"])
    os.environ.setdefault(
        "HF_HOME", str(Path(cfg["paths"]["base_model_dir"]) / "huggingface")
    )
    return cfg


def path_for(cfg: Dict[str, Any], section: str, key: str) -> Path:
    return Path(cfg[section][key])

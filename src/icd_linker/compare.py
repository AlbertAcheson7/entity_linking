from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .io_utils import write_json


def compare(cfg: Dict[str, Any]) -> Dict[str, Any]:
    root = Path(cfg["paths"]["experiments_dir"])
    results = {}
    for variant in ("base", "finetuned"):
        path = root / variant / "test_reranked_metrics.json"
        if not path.exists():
            path = root / variant / "test_retrieval_metrics.json"
        results[variant] = json.loads(path.read_text(encoding="utf-8"))
    base = results["base"]["metrics"]
    fine = results["finetuned"]["metrics"]
    if base.get("queries") != fine.get("queries"):
        raise RuntimeError("baseline and finetuned results use different test sets")
    deltas = {
        key: fine[key] - base[key]
        for key in base if isinstance(base[key], (float, int)) and key != "queries"
    }
    output = {"results": results, "finetuned_minus_base": deltas}
    write_json(root / "comparison.json", output)
    return output

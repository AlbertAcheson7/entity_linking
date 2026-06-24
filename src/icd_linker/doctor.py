from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path
from typing import Any, Dict


def doctor(cfg: Dict[str, Any]) -> Dict[str, Any]:
    packages = {
        name: importlib.util.find_spec(name) is not None
        for name in (
            "torch", "transformers", "chromadb", "BCEmbedding", "yaml",
            "numpy",
        )
    }
    try:
        import torch

        cuda = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda else ""
        gpu_memory_gib = (
            torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            if cuda else 0.0
        )
    except Exception:
        cuda, gpu_name, gpu_memory_gib = False, "", 0.0
    source_dir = Path(cfg["paths"]["source_dir"])
    files = {
        key: {
            "path": str(source_dir / filename),
            "exists": (source_dir / filename).exists(),
            "size_bytes": (
                (source_dir / filename).stat().st_size
                if (source_dir / filename).exists() else 0
            ),
        }
        for key, filename in cfg["input_files"].items()
    }
    disk = shutil.disk_usage(Path(cfg["_root"]))
    result = {
        "packages": packages,
        "cuda_available": cuda,
        "gpu_name": gpu_name,
        "gpu_memory_gib": round(gpu_memory_gib, 2),
        "source_files": files,
        "free_disk_gib": round(disk.free / 1024 ** 3, 2),
        "hf_endpoint": cfg["models"]["hf_endpoint"],
        "passed": (
            all(packages.values())
            and cuda
            and gpu_memory_gib >= 20
            and all(item["exists"] for item in files.values())
            and disk.free >= 20 * 1024 ** 3
        ),
    }
    return result

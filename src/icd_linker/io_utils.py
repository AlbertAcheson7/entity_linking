from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List


def read_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: {exc}") from exc


def write_jsonl(path: str | Path, records: Iterable[Dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    os.replace(tmp, path)
    return count


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(tmp, path)


def load_lookup(path: str | Path, key: str = "term_uid") -> Dict[str, dict]:
    return {record[key]: record for record in read_jsonl(path)}


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def seeded_split(
    ids: List[str], train: float, validation: float, seed: int
) -> Dict[str, set[str]]:
    ids = sorted(set(ids))
    random.Random(seed).shuffle(ids)
    n = len(ids)
    train_end = int(n * train)
    validation_end = train_end + int(n * validation)
    return {
        "train": set(ids[:train_end]),
        "validation": set(ids[train_end:validation_end]),
        "test": set(ids[validation_end:]),
    }


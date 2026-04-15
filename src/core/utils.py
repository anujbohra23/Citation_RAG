from pathlib import Path
import json
from typing import Iterable, Dict, Any


def ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_parent_dir(path)
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)

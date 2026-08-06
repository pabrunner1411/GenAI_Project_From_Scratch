import json
from pathlib import Path
from typing import Any


def read_text(file_path: str | Path) -> str:
    return Path(file_path).read_text(encoding="utf-8")


def read_json(file_path: str | Path) -> Any:
    return json.loads(Path(file_path).read_text(encoding="utf-8"))


def write_json(data: Any, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=4),
        encoding="utf-8",
    )

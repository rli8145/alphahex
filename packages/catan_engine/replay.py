from __future__ import annotations

import json
from pathlib import Path
from time import time


DEFAULT_REPLAY_DIR = Path("data/replays")


def save_replay(result: dict, path: str | None = None) -> str:
    replay_dir = DEFAULT_REPLAY_DIR
    replay_dir.mkdir(parents=True, exist_ok=True)
    if path is None:
        seed = result.get("seed", 0)
        stamp = int(time() * 1000)
        target = replay_dir / f"replay_seed_{seed}_{stamp}.json"
    else:
        target = Path(path)
        if not target.is_absolute():
            target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return str(target)


def list_replays() -> list[str]:
    DEFAULT_REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(path.name for path in DEFAULT_REPLAY_DIR.glob("*.json"))


def load_replay(filename: str) -> dict:
    safe_name = Path(filename).name
    path = DEFAULT_REPLAY_DIR / safe_name
    return json.loads(path.read_text(encoding="utf-8"))

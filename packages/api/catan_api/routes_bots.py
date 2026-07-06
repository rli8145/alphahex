from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from fastapi import APIRouter

from catan_bots import available_bots
from catan_bots.value_network import DEFAULT_VALUE_NETWORK_PATH, checkpoint_serving_ready

router = APIRouter()


@router.get("/bots")
def bots() -> dict:
    return {"bots": available_bots()}


@router.get("/bots/version")
def bot_version() -> dict:
    checkpoint = DEFAULT_VALUE_NETWORK_PATH
    if not checkpoint.exists():
        return {
            "bot": "mcts",
            "active_model": "heuristic",
            "version": "heuristic-baseline",
            "serving_ready": False,
            "checkpoint": None,
            "updated_at": None,
            "training": {},
        }

    raw = checkpoint.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()[:12]
    try:
        payload = json.loads(raw.decode("utf-8"))
        training = payload.get("training", {}) if isinstance(payload, dict) else {}
        if not isinstance(training, dict):
            training = {}
    except (UnicodeDecodeError, json.JSONDecodeError):
        training = {}

    serving_ready = checkpoint_serving_ready(checkpoint)
    updated_at = datetime.fromtimestamp(checkpoint.stat().st_mtime, timezone.utc).isoformat()
    return {
        "bot": "mcts",
        "active_model": "neural" if serving_ready else "heuristic",
        "version": f"nn-{digest}" if serving_ready else "heuristic-baseline",
        "checkpoint_version": digest,
        "serving_ready": serving_ready,
        "checkpoint": str(checkpoint),
        "updated_at": updated_at,
        "training": {
            "accepted": training.get("accepted"),
            "baseline_gate_passed": training.get("baseline_gate_passed"),
            "saved_network": training.get("saved_network"),
            "hidden_size": training.get("hidden_size"),
            "ml_framework": training.get("ml_framework"),
            "torch_device": training.get("torch_device"),
            "seed": training.get("seed"),
            "games": training.get("games"),
            "board_rule_version": training.get("board_rule_version"),
        },
    }

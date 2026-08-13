from __future__ import annotations

from fastapi import APIRouter, HTTPException

from catan_engine.replay import list_replays, load_replay

router = APIRouter()


@router.get("/replays")
def replays() -> dict:
    return {"replays": list_replays()}


@router.get("/replays/{filename}")
def replay(filename: str) -> dict:
    try:
        return load_replay(filename)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="replay not found") from exc

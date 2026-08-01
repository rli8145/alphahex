"""Optional Postgres persistence for completed games.

Configured entirely via the DATABASE_URL (or POSTGRES_URL) env var - Vercel's
Postgres (Neon) integration injects one of these automatically once added to
the project from the dashboard's Storage tab (see README). Without either set
(e.g. local dev with no database running), every function here is a no-op, so
gameplay works exactly as it did before this module existed.

Schema is applied with CREATE TABLE IF NOT EXISTS on every call instead of a
startup hook, since it's not guaranteed a lifespan event fires on Vercel's
per-request serverless invocation - the IF NOT EXISTS check is cheap enough
at this project's traffic to not bother with a real migration step.
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    id BIGSERIAL PRIMARY KEY,
    seed BIGINT NOT NULL,
    winner SMALLINT NOT NULL,
    turn_count INTEGER NOT NULL,
    move_count INTEGER NOT NULL,
    user_id TEXT,
    finished_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
ALTER TABLE games ADD COLUMN IF NOT EXISTS user_id TEXT;
"""


def _database_url() -> str | None:
    return os.environ.get("DATABASE_URL") or os.environ.get("POSTGRES_URL")


def _connect():
    import psycopg  # lazy import: the module (and app) must still load with no driver/DB configured

    return psycopg.connect(_database_url(), connect_timeout=5)


def record_finished_game(
    *, seed: int, winner: int, turn_count: int, move_count: int, user_id: str | None = None
) -> None:
    """Record one completed game. Best-effort - a DB hiccup must never break a game response.

    user_id is the Supabase auth uid (see catan_api.auth.current_user), or None for an
    anonymous player - games record either way, only the attribution differs.
    """
    if not _database_url():
        return
    try:
        with _connect() as conn:
            conn.execute(_SCHEMA)
            conn.execute(
                "INSERT INTO games (seed, winner, turn_count, move_count, user_id) VALUES (%s, %s, %s, %s, %s)",
                (seed, winner, turn_count, move_count, user_id),
            )
    except Exception:
        logger.warning("failed to record finished game", exc_info=True)


def fetch_recent_games(limit: int = 20, *, user_id: str | None = None) -> list[dict[str, Any]]:
    """Most recent completed games, optionally restricted to one user_id."""
    if not _database_url():
        return []
    try:
        with _connect() as conn:
            conn.execute(_SCHEMA)
            query = "SELECT id, seed, winner, turn_count, move_count, user_id, finished_at FROM games "
            params: tuple[Any, ...] = ()
            if user_id is not None:
                query += "WHERE user_id = %s "
                params = (user_id,)
            query += "ORDER BY finished_at DESC LIMIT %s"
            rows = conn.execute(query, (*params, limit)).fetchall()
        return [
            {
                "id": row[0],
                "seed": row[1],
                "winner": row[2],
                "turn_count": row[3],
                "move_count": row[4],
                "user_id": row[5],
                "finished_at": row[6].isoformat(),
            }
            for row in rows
        ]
    except Exception:
        logger.warning("failed to fetch game history", exc_info=True)
        return []

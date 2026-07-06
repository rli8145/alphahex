"""Vercel serverless entrypoint for the Catan FastAPI backend.

The repo keeps its Python packages under packages/*, which are normally put on
PYTHONPATH; on Vercel we add them to sys.path here instead.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for package in ("engine", "bots", "api"):
    sys.path.insert(0, str(ROOT / "packages" / package))

from catan_api.app import app as catan_app  # noqa: E402


async def app(scope, receive, send):
    # Vercel rewrites /api/(.*) to this function with the original URL intact,
    # but the FastAPI routes are defined without the /api prefix — strip it.
    if scope["type"] == "http" and scope["path"].startswith("/api"):
        scope = dict(scope)
        scope["path"] = scope["path"][len("/api"):] or "/"
        scope["raw_path"] = scope["path"].encode()
    await catan_app(scope, receive, send)

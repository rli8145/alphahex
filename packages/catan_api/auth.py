"""Optional Supabase auth: resolve the caller's user id from a bearer JWT.

Supabase signs JWTs asymmetrically (JWKS) - the default for every project
created since late 2025 - so verification here fetches the project's public
signing keys instead of checking against a shared secret. Configured via
SUPABASE_URL (same value as the frontend's VITE_SUPABASE_URL); without it
set, or with no/invalid Authorization header, current_user() returns None
and every route keeps working exactly as it did before this module existed -
fully anonymous, nothing tied to a user.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache

from fastapi import Header

logger = logging.getLogger(__name__)

# Supabase only ever signs with an asymmetric algorithm under this scheme.
# HS* must never be added here - accepting it alongside an asymmetric key
# would open the classic alg-confusion attack (a forged HS-signed token
# verified against a key meant only for asymmetric use).
_ALGORITHMS = ["ES256"]


def _supabase_url() -> str | None:
    return os.environ.get("SUPABASE_URL")


@lru_cache(maxsize=1)
def _jwk_client(supabase_url: str):
    import jwt  # lazy import: this module (and the app) must still load with no PyJWT-needing config

    return jwt.PyJWKClient(f"{supabase_url}/auth/v1/.well-known/jwks.json", cache_keys=True, timeout=5)


def current_user(authorization: str | None = Header(default=None)) -> str | None:
    """Return the caller's Supabase user id (a uuid) if a valid bearer token was sent, else None."""
    supabase_url = _supabase_url()
    if not supabase_url or not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization.removeprefix("Bearer ")
    try:
        import jwt

        signing_key = _jwk_client(supabase_url).get_signing_key_from_jwt(token)
        payload = jwt.decode(token, signing_key.key, algorithms=_ALGORITHMS, audience="authenticated")
        return payload.get("sub")
    except Exception:
        logger.warning("rejected invalid Supabase auth token", exc_info=True)
        return None

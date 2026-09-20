"""Access control for the API.

The server spends your TypeSafe and Gemini quota on every question, so an
unauthenticated deployment is an open tap. Set VERBATIM_ACCESS_TOKEN and every
protected route requires it as a bearer token.

- Locally, with no token set, the API is open (it only listens on localhost).
- On Vercel (the platform sets VERCEL), a missing token *fails closed*: the API
  refuses to serve rather than silently exposing your keys' budget.

The token is compared in constant time. The static page itself is public; it
holds no secrets and can't do anything without the token.
"""

from __future__ import annotations

import os
import secrets
from typing import Annotated

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

_bearer = HTTPBearer(auto_error=False, description="The VERBATIM_ACCESS_TOKEN value")


def access_token() -> str | None:
    return os.environ.get("VERBATIM_ACCESS_TOKEN") or None


def auth_required() -> bool:
    """Whether callers must present a token (always true on Vercel)."""
    return access_token() is not None or bool(os.environ.get("VERCEL"))


def require_access(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> None:
    expected = access_token()
    if expected is None:
        if os.environ.get("VERCEL"):
            raise HTTPException(
                status_code=503,
                detail="The server has no access token configured: set VERBATIM_ACCESS_TOKEN.",
            )
        return  # local development

    supplied = credentials.credentials if credentials else ""
    if not secrets.compare_digest(supplied.encode(), expected.encode()):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid access token.",
            headers={"WWW-Authenticate": "Bearer"},
        )


AccessDep = Depends(require_access)

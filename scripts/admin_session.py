"""Admin login helper for scripts that call admin-only API endpoints.

/api/v1/answer and /api/v1/search require an administrator session. Set
LAWCHAT_ADMIN_USERNAME and LAWCHAT_ADMIN_PASSWORD in the environment; the
password is never read from or written to a file by these scripts.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from http.cookies import SimpleCookie

SESSION_COOKIE = "lawchat_session"


def base_url_of(endpoint_url: str) -> str:
    """'http://host:8000/api/v1/answer' -> 'http://host:8000'."""
    return endpoint_url.split("/api/", 1)[0].rstrip("/")


def admin_cookies(base_url: str) -> dict[str, str]:
    username = os.getenv("LAWCHAT_ADMIN_USERNAME")
    password = os.getenv("LAWCHAT_ADMIN_PASSWORD")
    if not username or not password:
        raise SystemExit(
            "Set LAWCHAT_ADMIN_USERNAME and LAWCHAT_ADMIN_PASSWORD: "
            "/api/v1/answer and /api/v1/search are admin-only."
        )
    request = urllib.request.Request(
        base_url.rstrip("/") + "/api/v1/auth/login",
        data=json.dumps({"username": username, "password": password}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            headers = response.headers.get_all("Set-Cookie") or []
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"Admin login failed: HTTP {exc.code}") from exc
    for header in headers:
        cookie = SimpleCookie(header)
        if SESSION_COOKIE in cookie:
            return {SESSION_COOKIE: cookie[SESSION_COOKIE].value}
    raise SystemExit("Admin login did not return a session cookie")


def cookie_header(cookies: dict[str, str]) -> str:
    return "; ".join(f"{name}={value}" for name, value in cookies.items())

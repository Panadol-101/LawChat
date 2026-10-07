"""Admin login helper for scripts that call admin-only API endpoints.

/api/v1/answer and /api/v1/search require an administrator session. Set
LAWCHAT_ADMIN_USERNAME, LAWCHAT_ADMIN_PASSWORD and either
LAWCHAT_ADMIN_TOTP_CODE (a 6-digit code from the authenticator app, used once
at login; the session then lasts 12 hours) or LAWCHAT_ADMIN_TOTP_SECRET (from
scripts/enroll_admin_totp.py) in the environment; these secrets are never read
from or written to a file by these scripts.
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


def _post(url: str, body: dict) -> tuple[dict, list[str]]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return (
                json.loads(response.read() or b"{}"),
                response.headers.get_all("Set-Cookie") or [],
            )
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"Admin login failed: HTTP {exc.code}") from exc


def admin_cookies(base_url: str) -> dict[str, str]:
    username = os.getenv("LAWCHAT_ADMIN_USERNAME")
    password = os.getenv("LAWCHAT_ADMIN_PASSWORD")
    if not username or not password:
        raise SystemExit(
            "Set LAWCHAT_ADMIN_USERNAME and LAWCHAT_ADMIN_PASSWORD: "
            "/api/v1/answer and /api/v1/search are admin-only."
        )
    base_url = base_url.rstrip("/")
    credentials = {"username": username, "password": password}
    payload, headers = _post(base_url + "/api/v1/auth/login", credentials)
    if payload.get("requires_totp"):
        # ADMIN accounts always need a second factor: either a one-time code
        # read from the authenticator app, or the secret to generate one.
        code = os.getenv("LAWCHAT_ADMIN_TOTP_CODE")
        secret = os.getenv("LAWCHAT_ADMIN_TOTP_SECRET")
        if not code and not secret:
            raise SystemExit(
                "Set LAWCHAT_ADMIN_TOTP_CODE (the 6-digit code from the authenticator app) "
                "or LAWCHAT_ADMIN_TOTP_SECRET (printed by scripts/enroll_admin_totp.py): "
                "administrator login requires TOTP."
            )
        if not code:
            import pyotp

            code = pyotp.TOTP(secret).now()

        _payload, headers = _post(
            base_url + "/api/v1/auth/totp",
            {**credentials, "code": code.strip()},
        )
    for header in headers:
        cookie = SimpleCookie(header)
        if SESSION_COOKIE in cookie:
            return {SESSION_COOKIE: cookie[SESSION_COOKIE].value}
    raise SystemExit("Admin login did not return a session cookie")


def cookie_header(cookies: dict[str, str]) -> str:
    return "; ".join(f"{name}={value}" for name, value in cookies.items())

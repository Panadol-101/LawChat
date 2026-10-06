from __future__ import annotations

import os
from typing import Any

import httpx


class APIError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code


class AdminAPI:
    def __init__(self) -> None:
        self.base_url = os.getenv(
            "LAWCHAT_API_BASE_URL",
            "http://127.0.0.1:8000",
        ).rstrip("/")

        self.session_token: str | None = None
        self.client = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(30, connect=5),
        )

    def close(self) -> None:
        self.client.close()

    def me(self) -> dict[str, Any]:
        return self._request("GET", "/api/v1/auth/me")

    def list_users(self, status: str | None = None) -> list[dict[str, Any]]:
        return self._request(
            "GET",
            "/api/v1/admin/users",
            params={"status": status} if status else None,
        )

    def approve_user(self, user_id: str) -> dict[str, Any]:
        return self._request("POST", f"/api/v1/admin/users/{user_id}/approve")

    def reject_user(self, user_id: str) -> dict[str, Any]:
        return self._request("POST", f"/api/v1/admin/users/{user_id}/reject")

    def delete_user(self, user_id: str) -> None:
        self._request("DELETE", f"/api/v1/admin/users/{user_id}")

    def update_user_usage(
        self,
        user_id: str,
        token_limit: int,
    ) -> dict[str, Any]:
        return self._request(
            "PATCH",
            f"/api/v1/admin/users/{user_id}/usage",
            json={
                "token_limit": token_limit,
            },
        )


    def login(
        self,
        username: str,
        password: str,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/api/v1/auth/login",
            json={
                "username": username,
                "password": password,
            },
        )

    def verify_totp(
        self,
        username: str,
        password: str,
        code: str,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/api/v1/auth/totp",
            json={
                "username": username,
                "password": password,
                "code": code,
            },
        )

    def logout(self) -> None:
        self._request("POST", "/api/v1/auth/logout")

    def _request(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> Any:
        try:
                response = self.client.request(
                    method,
                    path,
                    **kwargs,
                )

                session_cookie = response.cookies.get("lawchat_session")

                if session_cookie:
                    self.client.cookies.set(
                        "lawchat_session",
                        session_cookie,
                        domain="127.0.0.1",
                        path="/",
                    )
        except httpx.TimeoutException as exc:
            raise APIError(
                "LawChat API phản hồi quá chậm."
            ) from exc
        except httpx.HTTPError as exc:
            raise APIError(
                "Không thể kết nối tới LawChat API."
            ) from exc

        if not response.is_success:
            try:
                detail = response.json().get("detail")
            except (ValueError, AttributeError):
                detail = None

            raise APIError(
                str(
                    detail
                    or f"LawChat API returned HTTP {response.status_code}"
                ),
                status_code=response.status_code,
            )

        if response.status_code == 204:
            return None

        return response.json()

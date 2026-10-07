from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Any

import httpx


REGISTER_SUCCESS_MESSAGE = (
    "Đăng ký thành công. Tài khoản của bạn đang chờ quản trị viên phê duyệt."
)

# Authentication/registration errors from the API, shown to users in Vietnamese.
# Other API messages (chat, projects, ...) are passed through unchanged.
AUTH_ERROR_MESSAGES = {
    "Invalid username or password": "Tên đăng nhập hoặc mật khẩu không chính xác.",
    "Account pending administrator approval": (
        "Tài khoản đang chờ quản trị viên phê duyệt."
    ),
    "Account has been rejected": "Tài khoản đã bị từ chối.",
    "Account is not active": "Tài khoản chưa được kích hoạt.",
    "Username already exists": "Tên đăng nhập đã tồn tại.",
    "Username is required": "Vui lòng nhập tên đăng nhập.",
    "Username must be between 3 and 50 characters": (
        "Tên đăng nhập phải có từ 3 đến 50 ký tự."
    ),
    "Passwords do not match": "Mật khẩu nhập lại không khớp.",
    "Too many failed login attempts; try again later": (
        "Bạn đã đăng nhập sai quá nhiều lần. Vui lòng thử lại sau."
    ),
    "Invalid authentication code": "Mã xác thực không chính xác.",
    "TOTP enrollment required for administrator accounts; "
    "run scripts/enroll_admin_totp.py": (
        "Tài khoản quản trị cần thiết lập xác thực hai lớp (TOTP) trước khi đăng nhập."
    ),
    "Authentication required": "Vui lòng đăng nhập để tiếp tục.",
    "Invalid or expired session": (
        "Phiên đăng nhập đã hết hạn. Vui lòng đăng nhập lại."
    ),
    "Administrator privileges required": "Bạn không có quyền thực hiện thao tác này.",
}

AUTH_PATH_PREFIX = "/api/v1/auth/"

# Request validation (HTTP 422) messages per auth form field.
AUTH_FIELD_MESSAGES = {
    "username": (
        "Tên đăng nhập phải có từ 3 đến 50 ký tự, chỉ gồm chữ cái không dấu, "
        "chữ số hoặc dấu gạch dưới (_)."
    ),
    "password": "Mật khẩu phải có từ 8 đến 128 ký tự.",
    "confirm_password": "Mật khẩu nhập lại phải có từ 8 đến 128 ký tự.",
    "code": "Mã xác thực phải gồm 6 chữ số.",
}
LOGIN_FIELD_MESSAGE = "Vui lòng nhập đầy đủ tên đăng nhập và mật khẩu hợp lệ."
INVALID_INPUT_MESSAGE = "Dữ liệu nhập không hợp lệ."


def user_error_message(status_code: int, detail: Any, path: str = "") -> str:
    """Message to show for a failed API call."""
    if isinstance(detail, str) and detail in AUTH_ERROR_MESSAGES:
        return AUTH_ERROR_MESSAGES[detail]

    if status_code == 422 and path.startswith(AUTH_PATH_PREFIX):
        return _validation_message(detail, path)

    if detail:
        return str(detail)
    return f"Máy chủ LawChat trả về lỗi (HTTP {status_code})."


def _validation_message(detail: Any, path: str) -> str:
    fields = []
    for error in detail if isinstance(detail, list) else []:
        loc = error.get("loc") if isinstance(error, dict) else None
        if loc:
            fields.append(str(loc[-1]))

    if path.endswith("/register"):
        messages = [AUTH_FIELD_MESSAGES[f] for f in fields if f in AUTH_FIELD_MESSAGES]
    elif path.endswith("/totp") and "code" in fields:
        messages = [AUTH_FIELD_MESSAGES["code"]]
    else:
        messages = [LOGIN_FIELD_MESSAGE]
    # Keep order, drop repeats.
    return " ".join(dict.fromkeys(messages)) or INVALID_INPUT_MESSAGE


class APIError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class LawChatAPI:
    def __init__(self) -> None:
        self.base_url = os.getenv(
            "LAWCHAT_API_BASE_URL", "http://127.0.0.1:8000"
        ).rstrip("/")
        self.workspace_id = os.getenv("LAWCHAT_WORKSPACE_ID", "local")
        self.client = httpx.Client(
            base_url=self.base_url,
            headers={"X-Workspace-ID": self.workspace_id},
            timeout=httpx.Timeout(330, connect=10),
        )

    def close(self) -> None:
        self.client.close()

    def health(self) -> bool:
        try:
            return self.client.get("/health", timeout=3).is_success
        except httpx.HTTPError:
            return False

    def list_projects(self) -> list[dict[str, Any]]:
        return self._request("GET", "/api/v1/projects")

    def create_project(self, name: str) -> dict[str, Any]:
        return self._request("POST", "/api/v1/projects", json={"name": name})

    def rename_project(self, project_id: str, name: str) -> dict[str, Any]:
        return self._request(
            "PATCH", f"/api/v1/projects/{project_id}", json={"name": name}
        )

    def delete_project(self, project_id: str) -> None:
        self._request("DELETE", f"/api/v1/projects/{project_id}")

    def list_conversations(
        self, project_id: str | None = None
    ) -> list[dict[str, Any]]:
        params = {"project_id": project_id} if project_id else None
        return self._request("GET", "/api/v1/conversations", params=params)

    def create_conversation(
        self, title: str, project_id: str | None = None
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/api/v1/conversations",
            json={"title": title, "project_id": project_id},
        )

    def get_conversation(self, conversation_id: str) -> dict[str, Any]:
        return self._request("GET", f"/api/v1/conversations/{conversation_id}")

    def update_conversation(
        self, conversation_id: str, **updates: Any
    ) -> dict[str, Any]:
        return self._request(
            "PATCH", f"/api/v1/conversations/{conversation_id}", json=updates
        )

    def delete_conversation(self, conversation_id: str) -> None:
        self._request("DELETE", f"/api/v1/conversations/{conversation_id}")

    def list_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        return self._request(
            "GET", f"/api/v1/conversations/{conversation_id}/messages"
        )

    def share_conversation(self, conversation_id: str) -> str:
        payload = self._request(
            "POST", f"/api/v1/conversations/{conversation_id}/share"
        )
        return payload["token"]

    def stop_sharing_conversation(self, conversation_id: str) -> None:
        self._request("DELETE", f"/api/v1/conversations/{conversation_id}/share")

    def get_shared_conversation(self, token: str) -> dict[str, Any]:
        return self._request("GET", f"/api/v1/shared/conversations/{token}")

    def stream_chat(
        self,
        *,
        conversation_id: str,
        client_message_id: str,
        message: str,
    ) -> Iterator[tuple[str, dict[str, Any]]]:
        payload = {
            "conversation_id": conversation_id,
            "client_message_id": client_message_id,
            "message": message,
            "response_mode": "compact",
        }
        try:
            with self.client.stream(
                "POST", "/api/v1/chat/stream", json=payload
            ) as response:
                self._raise_for_status(response)
                event_name = "message"
                data_lines: list[str] = []
                for line in response.iter_lines():
                    if line.startswith(":"):
                        continue
                    if not line:
                        if data_lines:
                            raw = "\n".join(data_lines)
                            try:
                                data = json.loads(raw)
                            except json.JSONDecodeError as exc:
                                raise APIError("Backend returned invalid SSE data") from exc
                            yield event_name, data
                        event_name = "message"
                        data_lines = []
                        continue
                    if line.startswith("event:"):
                        event_name = line[6:].strip()
                    elif line.startswith("data:"):
                        data_lines.append(line[5:].lstrip())
                if data_lines:
                    yield event_name, json.loads("\n".join(data_lines))
        except httpx.TimeoutException as exc:
            raise APIError("LawChat xử lý quá thời gian cho phép.") from exc
        except httpx.HTTPError as exc:
            raise APIError("Không thể kết nối tới LawChat API.") from exc

    def register(
        self,
        username: str,
        password: str,
        confirm_password: str,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/api/v1/auth/register",
            json={
                "username": username,
                "password": password,
                "confirm_password": confirm_password,
            },
        )

    def login(self, username: str, password: str) -> dict[str, Any]:
        return self._request(
            "POST",
            "/api/v1/auth/login",
            json={"username": username, "password": password},
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

    def me(self) -> dict[str, Any]:
        return self._request("GET", "/api/v1/auth/me")

    def logout(self) -> None:
        self._request("POST", "/api/v1/auth/logout")

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = self.client.request(method, path, **kwargs)
            session_cookie = response.cookies.get("lawchat_session")
            if session_cookie:
                self.client.cookies.set(
                    "lawchat_session",
                    session_cookie,
                    domain="127.0.0.1",
                    path="/",
                )
        except httpx.TimeoutException as exc:
            raise APIError("LawChat API phản hồi quá chậm.") from exc
        except httpx.HTTPError as exc:
            raise APIError("Không thể kết nối tới LawChat API.") from exc
        self._raise_for_status(response)
        if response.status_code == 204:
            return None
        return response.json()

    
    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        if response.is_success:
           return

        try:
            response.read()
            detail = response.json().get("detail")
        except (ValueError, AttributeError, httpx.ResponseNotRead):
            detail = None

        raise APIError(
           user_error_message(
               response.status_code, detail, _request_path(response)
           ),
           status_code=response.status_code,
        )


def _request_path(response: httpx.Response) -> str:
    try:
        return response.request.url.path
    except RuntimeError:
        return ""

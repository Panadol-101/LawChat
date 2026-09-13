from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Any

import httpx


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

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            response = self.client.request(method, path, **kwargs)
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
            detail = response.json().get("detail")
        except (ValueError, AttributeError):
            detail = None
        raise APIError(
            str(detail or f"LawChat API returned HTTP {response.status_code}"),
            status_code=response.status_code,
        )

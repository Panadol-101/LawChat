"""Public frontend shows Vietnamese for authentication/registration errors."""

import httpx
import pytest

import api.main as main
from auth.service import USERNAME_EXISTS
from frontend.api_client import (
    AUTH_ERROR_MESSAGES,
    REGISTER_SUCCESS_MESSAGE,
    APIError,
    LawChatAPI,
    user_error_message,
)


def _raise(status_code, detail, path):
    response = httpx.Response(
        status_code,
        json={"detail": detail},
        request=httpx.Request("POST", f"http://api{path}"),
    )
    with pytest.raises(APIError) as exc_info:
        LawChatAPI._raise_for_status(response)
    return str(exc_info.value)


@pytest.mark.parametrize(
    ("detail", "expected"),
    [
        ("Account has been rejected", "Tài khoản đã bị từ chối."),
        ("Account pending administrator approval",
         "Tài khoản đang chờ quản trị viên phê duyệt."),
        ("Username already exists", "Tên đăng nhập đã tồn tại."),
        ("Invalid username or password",
         "Tên đăng nhập hoặc mật khẩu không chính xác."),
    ],
)
def test_required_translations(detail, expected):
    assert user_error_message(403, detail, "/api/v1/auth/login") == expected


def test_every_backend_auth_message_has_a_translation():
    backend_messages = [
        *main.ACCOUNT_NOT_ACTIVE_DETAIL.values(),
        main.TOTP_ENROLLMENT_REQUIRED,
        USERNAME_EXISTS,
    ]

    for message in backend_messages:
        assert message in AUTH_ERROR_MESSAGES


def test_rejected_login_error_is_vietnamese():
    message = _raise(403, "Account has been rejected", "/api/v1/auth/login")

    assert message == "Tài khoản đã bị từ chối."


def test_pending_login_error_is_vietnamese():
    message = _raise(403, "Account pending administrator approval", "/api/v1/auth/totp")

    assert message == "Tài khoản đang chờ quản trị viên phê duyệt."


def test_username_conflict_is_vietnamese():
    assert _raise(400, "Username already exists", "/api/v1/auth/register") == (
        "Tên đăng nhập đã tồn tại."
    )


def test_registration_success_is_vietnamese():
    assert REGISTER_SUCCESS_MESSAGE == (
        "Đăng ký thành công. Tài khoản của bạn đang chờ quản trị viên phê duyệt."
    )


def test_register_validation_errors_are_vietnamese():
    detail = [
        {"type": "string_pattern_mismatch", "loc": ["body", "username"], "msg": "..."},
        {"type": "string_too_short", "loc": ["body", "password"], "msg": "..."},
    ]

    message = _raise(422, detail, "/api/v1/auth/register")

    assert message == (
        "Tên đăng nhập phải có từ 3 đến 50 ký tự, chỉ gồm chữ cái không dấu, "
        "chữ số hoặc dấu gạch dưới (_). Mật khẩu phải có từ 8 đến 128 ký tự."
    )


def test_login_validation_error_is_vietnamese():
    detail = [{"type": "missing", "loc": ["body", "password"], "msg": "Field required"}]

    assert _raise(422, detail, "/api/v1/auth/login") == (
        "Vui lòng nhập đầy đủ tên đăng nhập và mật khẩu hợp lệ."
    )


def test_missing_detail_falls_back_to_vietnamese():
    response = httpx.Response(
        502, text="bad gateway", request=httpx.Request("GET", "http://api/x")
    )

    with pytest.raises(APIError) as exc_info:
        LawChatAPI._raise_for_status(response)

    assert str(exc_info.value) == "Máy chủ LawChat trả về lỗi (HTTP 502)."


def test_non_auth_messages_are_unchanged():
    # Chat/project errors are outside this translation layer.
    assert user_error_message(404, "conversation not found", "/api/v1/conversations/x") == (
        "conversation not found"
    )

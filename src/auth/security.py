from __future__ import annotations

import hashlib
import secrets

import pyotp
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError


_password_hasher = PasswordHasher()


def hash_password(password: str) -> str:
    """Hash a plaintext password using Argon2id."""
    if not password:
        raise ValueError("Password must not be empty")

    return _password_hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a plaintext password against an Argon2 hash."""
    if not password or not password_hash:
        return False

    try:
        return _password_hasher.verify(password_hash, password)
    except (InvalidHashError, VerificationError):
        return False


def generate_session_token() -> str:
    """Generate a cryptographically secure opaque session token."""
    return secrets.token_urlsafe(32)


def hash_session_token(token: str) -> str:
    """Hash a session token before storing it in the database."""
    if not token:
        raise ValueError("Session token must not be empty")

    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def generate_totp_secret() -> str:
    """Generate a new Base32 TOTP secret."""
    return pyotp.random_base32()


def verify_totp(secret: str, code: str) -> bool:
    """Verify a TOTP code."""
    if not secret or not code:
        return False

    normalized_code = code.strip().replace(" ", "")

    if len(normalized_code) != 6 or not normalized_code.isdigit():
        return False

    return pyotp.TOTP(secret).verify(
        normalized_code,
        valid_window=1,
    )

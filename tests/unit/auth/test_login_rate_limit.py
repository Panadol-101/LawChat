from __future__ import annotations

from datetime import datetime

import pyotp

from auth.rate_limit import LoginRateLimiter
from auth.security import match_totp_step, verify_totp


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _limiter(clock):
    return LoginRateLimiter(
        max_attempts=3, window_seconds=60, lockout_seconds=120, clock=clock
    )


def test_lockout_after_max_failures_and_expiry():
    clock = FakeClock()
    limiter = _limiter(clock)
    keys = limiter.keys("10.0.0.1", "Admin")

    assert limiter.record_failure(keys) is False
    assert limiter.record_failure(keys) is False
    assert limiter.record_failure(keys) is True
    assert limiter.retry_after(keys) == 120

    clock.now += 121
    assert limiter.retry_after(keys) == 0


def test_failures_outside_window_do_not_count():
    clock = FakeClock()
    limiter = _limiter(clock)
    keys = limiter.keys(None, "bob")

    limiter.record_failure(keys)
    limiter.record_failure(keys)
    clock.now += 61
    assert limiter.record_failure(keys) is False
    assert limiter.retry_after(keys) == 0


def test_username_lock_applies_from_any_ip_and_ignores_case():
    clock = FakeClock()
    limiter = _limiter(clock)
    for ip in ("1.1.1.1", "2.2.2.2", "3.3.3.3"):
        limiter.record_failure(limiter.keys(ip, "admin"))

    assert limiter.retry_after(limiter.keys("4.4.4.4", "ADMIN ")) > 0


def test_success_clears_username_but_not_ip():
    clock = FakeClock()
    limiter = _limiter(clock)
    keys = limiter.keys("9.9.9.9", "bob")
    limiter.record_failure(keys)
    limiter.record_failure(keys)

    limiter.record_success("bob")

    assert limiter.record_failure(limiter.keys(None, "bob")) is False
    assert limiter.record_failure(limiter.keys("9.9.9.9", "carol")) is True


def test_match_totp_step_returns_current_step():
    secret = pyotp.random_base32()
    totp = pyotp.TOTP(secret)

    assert match_totp_step(secret, totp.now()) == totp.timecode(datetime.now())
    assert match_totp_step(secret, "000000x") is None
    assert verify_totp(secret, "12345") is False


def test_loopback_logins_are_limited_per_username_only():
    limiter = _limiter(FakeClock())
    for _ in range(3):
        limiter.record_failure(limiter.keys("127.0.0.1", "typo-user"))

    assert limiter.retry_after(limiter.keys("127.0.0.1", "typo-user")) > 0
    assert limiter.retry_after(limiter.keys("127.0.0.1", "someone-else")) == 0

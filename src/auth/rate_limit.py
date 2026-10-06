from __future__ import annotations

import ipaddress
import os
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class LoginRateLimiter:
    """Sliding-window limit on failed logins, keyed by client IP and by username.

    State is in process memory: it resets on restart and is only exact with a
    single uvicorn worker (the current deployment). Move it to PostgreSQL or
    Redis before running several workers.
    """

    max_attempts: int = 5
    window_seconds: float = 15 * 60
    lockout_seconds: float = 15 * 60
    clock: Callable[[], float] = time.monotonic
    _failures: dict[str, deque[float]] = field(default_factory=dict)
    _locked_until: dict[str, float] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @classmethod
    def from_env(cls) -> LoginRateLimiter:
        return cls(
            max_attempts=int(os.getenv("LAWCHAT_LOGIN_MAX_ATTEMPTS", "5")),
            window_seconds=float(os.getenv("LAWCHAT_LOGIN_WINDOW_SECONDS", "900")),
            lockout_seconds=float(os.getenv("LAWCHAT_LOGIN_LOCKOUT_SECONDS", "900")),
        )

    @staticmethod
    def keys(ip_address: str | None, username: str) -> tuple[str, ...]:
        keys = [f"user:{username.strip().casefold()}"]
        # The Streamlit frontends call the API over loopback (host networking), so
        # every browser login shares 127.0.0.1; counting it would let one user's
        # typos lock everyone out. Such logins are limited per username only.
        if ip_address and not _is_loopback(ip_address):
            keys.append(f"ip:{ip_address}")
        return tuple(keys)

    def retry_after(self, keys: tuple[str, ...]) -> int:
        """Seconds until the caller may try again; 0 when not locked."""
        now = self.clock()
        with self._lock:
            remaining = max(
                (self._locked_until.get(key, 0.0) - now for key in keys),
                default=0.0,
            )
        return max(int(remaining + 0.999), 0)

    def record_failure(self, keys: tuple[str, ...]) -> bool:
        """Count one failed attempt; return True when it triggers a lockout."""
        now = self.clock()
        locked = False
        with self._lock:
            for key in keys:
                failures = self._failures.setdefault(key, deque())
                failures.append(now)
                while failures and failures[0] <= now - self.window_seconds:
                    failures.popleft()
                if len(failures) >= self.max_attempts:
                    self._locked_until[key] = now + self.lockout_seconds
                    failures.clear()
                    locked = True
        return locked

    def record_success(self, username: str) -> None:
        key = f"user:{username.strip().casefold()}"
        with self._lock:
            self._failures.pop(key, None)
            self._locked_until.pop(key, None)


def _is_loopback(ip_address: str) -> bool:
    try:
        return ipaddress.ip_address(ip_address).is_loopback
    except ValueError:
        return False

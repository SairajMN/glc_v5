"""Per-(channel, channel_user_id) rate limiting.

Sliding 60s windows for both messages_per_minute and tool_calls_per_minute.
Limits are read from channels.yaml's `defaults.rate_limits` block and may
be overridden per channel.

The interceptor sits *before* the policy engine so a rate-limited call
short-circuits to 429 without consuming any policy or LLM budget.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class _Window:
    messages: deque[float] = field(default_factory=deque)
    tool_calls: deque[float] = field(default_factory=deque)


def _gc(dq: deque[float], horizon: float) -> None:
    while dq and dq[0] < horizon:
        dq.popleft()


# How often _check pauses to drop windows that have aged out entirely.
_SWEEP_INTERVAL_SECONDS = 60.0


class RateLimiter:
    def __init__(self, default_mpm: int = 30, default_tpm: int = 20) -> None:
        self.default_mpm = default_mpm
        self.default_tpm = default_tpm
        self.per_channel: dict[str, dict[str, int]] = {}
        self._state: dict[tuple[str, str], _Window] = {}
        self._lock = threading.Lock()
        self._last_sweep = 0.0

    def configure_from_yaml(self, channels_yaml: dict) -> None:
        defaults = (channels_yaml or {}).get("defaults", {}).get("rate_limits", {})
        self.default_mpm = int(defaults.get("messages_per_minute", self.default_mpm))
        self.default_tpm = int(defaults.get("tool_calls_per_minute", self.default_tpm))
        for ch, cfg in ((channels_yaml or {}).get("channels", {}) or {}).items():
            rl = (cfg or {}).get("rate_limits") or {}
            if rl:
                self.per_channel[ch] = {
                    "messages_per_minute": int(rl.get("messages_per_minute", self.default_mpm)),
                    "tool_calls_per_minute": int(rl.get("tool_calls_per_minute", self.default_tpm)),
                }

    def limits_for(self, channel: str) -> tuple[int, int]:
        cfg = self.per_channel.get(channel)
        if cfg:
            return cfg["messages_per_minute"], cfg["tool_calls_per_minute"]
        return self.default_mpm, self.default_tpm

    def check_message(self, channel: str, user_id: str) -> tuple[bool, str]:
        return self._check(channel, user_id, "messages")

    def check_tool_call(self, channel: str, user_id: str) -> tuple[bool, str]:
        return self._check(channel, user_id, "tool_calls")

    def _sweep_locked(self, now: float) -> None:
        """Drop windows whose timestamps have all aged out.

        ``_gc`` only trims *inside* a window; nothing ever removed the window
        itself, so one ``_Window`` per distinct ``(channel, user_id)`` was
        retained for the life of the process. A sender who rotates
        ``channel_user_id`` therefore grows ``_state`` without bound, which is
        a memory exhaustion path that the per-identity cap cannot see, since
        every rotated id looks like a brand new caller under its own limit.

        Callers hold ``self._lock``.
        """
        if now - self._last_sweep < _SWEEP_INTERVAL_SECONDS:
            return
        self._last_sweep = now
        horizon = now - 60
        dead = []
        for key, win in self._state.items():
            _gc(win.messages, horizon)
            _gc(win.tool_calls, horizon)
            if not win.messages and not win.tool_calls:
                dead.append(key)
        for key in dead:
            del self._state[key]

    def _check(self, channel: str, user_id: str, kind: str) -> tuple[bool, str]:
        mpm, tpm = self.limits_for(channel)
        cap = mpm if kind == "messages" else tpm
        with self._lock:
            now = time.time()
            self._sweep_locked(now)
            win = self._state.setdefault((channel, user_id), _Window())
            dq = win.messages if kind == "messages" else win.tool_calls
            _gc(dq, now - 60)
            if len(dq) >= cap:
                return False, f"{kind} limit {cap}/min exceeded for ({channel}, {user_id})"
            dq.append(now)
            return True, ""


_limiter: RateLimiter | None = None


def get_rate_limiter() -> RateLimiter:
    global _limiter
    if _limiter is None:
        from glc.config import load_channels

        _limiter = RateLimiter()
        _limiter.configure_from_yaml(load_channels())
    return _limiter

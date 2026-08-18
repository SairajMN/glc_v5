"""Rate limiter key space must be bounded (backport glc_v2 #102).

_gc trims the timestamps inside a window. Nothing removed the window itself,
so one _Window per distinct (channel, user_id) survived for the life of the
process. Rotating channel_user_id therefore grows _state without bound, and
the per-identity cap cannot see it: every rotated id looks like a new caller
under its own limit.
"""

from __future__ import annotations

import glc.security.rate_limits as rl_mod
from glc.security.rate_limits import RateLimiter


def test_idle_windows_are_evicted(monkeypatch):
    """500 rotated identities must not leave 500 windows behind forever."""
    rl = RateLimiter(default_mpm=30, default_tpm=20)

    now = [1_000_000.0]
    monkeypatch.setattr(rl_mod.time, "time", lambda: now[0])

    for i in range(500):
        assert rl.check_message("telegram", f"rotated-{i}")[0]
    assert len(rl._state) == 500, "each fresh identity should get its own window"

    # Every timestamp is now well past the 60s horizon, so none of these
    # windows can still affect any decision. One more request gives the
    # limiter the chance to notice that.
    now[0] += 3600
    rl.check_message("telegram", "someone-new")

    assert len(rl._state) <= 2, f"idle windows were retained: {len(rl._state)}"


def test_sweep_keeps_a_window_that_is_still_live(monkeypatch):
    """A sweep must not drop an identity whose minute has not elapsed.

    Eviction that also forgot live callers would silently reset the cap, so
    this pins the difference between "aged out" and "inconvenient".
    """
    rl = RateLimiter(default_mpm=5, default_tpm=5)

    now = [2_000_000.0]
    monkeypatch.setattr(rl_mod.time, "time", lambda: now[0])

    for _ in range(5):
        assert rl.check_message("telegram", "steady")[0]

    # Force the next call to sweep while "steady" still has fresh timestamps.
    rl._last_sweep = now[0] - 10_000
    ok, msg = rl.check_message("telegram", "steady")
    assert ok is False, f"live identity lost its window to the sweep: {msg}"
    assert ("telegram", "steady") in rl._state


def test_limit_still_enforced_and_resets_normally(monkeypatch):
    """The cap must survive eviction, and still reset once the minute passes."""
    rl = RateLimiter(default_mpm=3, default_tpm=3)

    now = [3_000_000.0]
    monkeypatch.setattr(rl_mod.time, "time", lambda: now[0])

    for _ in range(3):
        assert rl.check_message("slack", "u1")[0]
    assert rl.check_message("slack", "u1")[0] is False

    now[0] += 61
    assert rl.check_message("slack", "u1")[0] is True

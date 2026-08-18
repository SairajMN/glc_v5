"""Caller metadata must not be able to relax the channel gate (backport glc_v2 #47/#108).

is_public_channel and was_mentioned were read straight out of env.metadata.
In allowed() they only ever make the check stricter, so the caller's lever is
to weaken it: claim is_public_channel=False and the public-channel mention
requirement never applies, or claim was_mentioned=True and it is satisfied
without a mention.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from glc.channels.envelope import ChannelMessage
from glc.routes import channels as ch_mod


def _env(text: str = "hello", **metadata) -> ChannelMessage:
    return ChannelMessage(
        channel="slack",
        channel_user_id="u-1",
        user_handle="someone",
        text=text,
        trust_level="user_paired",
        arrived_at=datetime.now(UTC),
        metadata=metadata,
    )


def _cfg(monkeypatch, **channel_cfg):
    monkeypatch.setattr(
        ch_mod,
        "load_channels",
        lambda: {"defaults": {"mention_only_in_public": True}, "channels": {"slack": channel_cfg}},
    )


def test_caller_cannot_downgrade_a_public_channel(monkeypatch):
    """Config says public; a caller claiming otherwise must not win."""
    _cfg(monkeypatch, is_public=True)
    is_public, _ = ch_mod._derive_gate("slack", _env(is_public_channel=False))
    assert is_public is True


def test_caller_admitting_public_is_honoured(monkeypatch):
    """Admitting a public channel is only ever stricter, so it is safe to take."""
    _cfg(monkeypatch)
    is_public, _ = ch_mod._derive_gate("slack", _env(is_public_channel=True))
    assert is_public is True


def test_claimed_mention_ignored_when_tokens_configured(monkeypatch):
    """With something to verify against, the caller's claim carries no weight."""
    _cfg(monkeypatch, is_public=True, mention_tokens=["@bot"])
    is_public, was_mentioned = ch_mod._derive_gate("slack", _env("hello all", was_mentioned=True))
    assert is_public is True
    assert was_mentioned is False, "a forged mention claim was believed"


def test_real_mention_is_detected(monkeypatch):
    _cfg(monkeypatch, is_public=True, mention_tokens=["@bot"])
    _, was_mentioned = ch_mod._derive_gate("slack", _env("hey @bot help", was_mentioned=False))
    assert was_mentioned is True, "a genuine mention was missed"


def test_without_tokens_behaviour_is_unchanged(monkeypatch):
    """No tokens configured means nothing to verify against; do not hard-deny."""
    _cfg(monkeypatch, is_public=True)
    _, was_mentioned = ch_mod._derive_gate("slack", _env("hello", was_mentioned=True))
    assert was_mentioned is True


def test_gate_is_never_looser_than_the_claim(monkeypatch):
    """The derived pair must not permit anything the raw claim would not."""
    _cfg(monkeypatch, is_public=True, mention_tokens=["@bot"])
    from glc.security.allowlists import allowed

    monkeypatch.setattr(
        "glc.security.allowlists.load_channels",
        lambda: {
            "defaults": {"mention_only_in_public": True},
            "channels": {"slack": {"allowed_senders": ["u-1"], "is_public": True}},
        },
    )
    env = _env("no mention here", is_public_channel=False, was_mentioned=True)
    gate_public, gate_mentioned = ch_mod._derive_gate("slack", env)
    ok, why = allowed(
        "slack", "u-1", owner_ids=[], is_public_channel=gate_public, was_mentioned=gate_mentioned
    )
    assert ok is False, f"forged metadata still passed the gate: {why}"


@pytest.mark.parametrize("claim", [True, False])
def test_derivation_is_stable_for_private_channels(monkeypatch, claim):
    """A private channel stays private regardless of what the caller says."""
    _cfg(monkeypatch, mention_tokens=["@bot"])
    is_public, _ = ch_mod._derive_gate("slack", _env(is_public_channel=claim))
    assert is_public is claim, "claim may only ever add strictness"

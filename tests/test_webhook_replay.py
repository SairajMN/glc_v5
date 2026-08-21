"""A signed webhook must be accepted once, not once per replay.

The freshness check bounds how long a captured request stays useful. It says
nothing about how many times it may be used inside that window, so a valid
body and signature could be replayed for five minutes, each replay producing
another inbound message, model call, reply and ledger entry.
"""

from __future__ import annotations

import hmac
import json
import time
from hashlib import sha256

import pytest

from glc.channels.catalogue.webhook.adapter import (
    REPLAY_WINDOW_SECONDS,
    Adapter,
    _SeenSignatures,
)

SECRET = "test-shared-secret"


def _signed(body: dict, ts: int | None = None) -> dict:
    raw = json.dumps(body, separators=(",", ":")).encode()
    ts = ts if ts is not None else int(time.time())
    signed = f"{ts}.{raw.decode()}".encode()
    mac = hmac.new(SECRET.encode(), signed, sha256).hexdigest()
    return {"raw_body": raw, "headers": {"X-Webhook-Signature": f"t={ts},v1={mac}"}}


@pytest.fixture(autouse=True)
def _secret(monkeypatch):
    monkeypatch.setenv("WEBHOOK_SHARED_SECRET", SECRET)


@pytest.mark.asyncio
async def test_same_signed_request_is_accepted_once():
    adapter = Adapter(config={})
    payload = _signed({"text": "hello", "sender_id": "u-1", "sender_handle": "someone"})

    first = await adapter.on_message(payload)
    assert first is not None, "a genuine first delivery must be accepted"

    second = await adapter.on_message(payload)
    assert second is None, "the identical request was replayed successfully"


@pytest.mark.asyncio
async def test_distinct_requests_still_pass():
    """Replay protection must not collapse two different genuine messages."""
    adapter = Adapter(config={})
    assert await adapter.on_message(_signed({"text": "one", "sender_id": "u-1", "sender_handle": "someone"})) is not None
    assert await adapter.on_message(_signed({"text": "two", "sender_id": "u-1", "sender_handle": "someone"})) is not None


@pytest.mark.asyncio
async def test_forged_signature_still_rejected():
    adapter = Adapter(config={})
    bad = _signed({"text": "hi", "sender_id": "u-1", "sender_handle": "someone"})
    bad["headers"]["X-Webhook-Signature"] = f"t={int(time.time())},v1=deadbeef"
    assert await adapter.on_message(bad) is None


@pytest.mark.asyncio
async def test_stale_timestamp_still_rejected():
    adapter = Adapter(config={})
    old = _signed({"text": "hi", "sender_id": "u-1", "sender_handle": "someone"}, ts=int(time.time()) - REPLAY_WINDOW_SECONDS - 60)
    assert await adapter.on_message(old) is None


def test_store_forgets_entries_past_the_window():
    """Entries stop mattering once freshness would reject them anyway."""
    store = _SeenSignatures(ttl_seconds=100)
    assert store.check_and_record("sig-a", now=1_000.0) is True
    assert store.check_and_record("sig-a", now=1_050.0) is False
    # Past the ttl the entry is swept, so the store cannot grow without bound.
    assert store.check_and_record("sig-b", now=1_500.0) is True
    assert len(store._seen) == 1


def test_store_records_only_what_it_is_given():
    store = _SeenSignatures(ttl_seconds=100)
    assert store.check_and_record("x", now=0.0) is True
    assert store.check_and_record("y", now=0.0) is True
    assert store.check_and_record("x", now=0.0) is False

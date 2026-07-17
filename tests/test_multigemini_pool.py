"""Offline proof that GLC v3 keeps Gemini keys as independently-metered
providers.  The live-graph scheduler must use this gateway seam, never keys."""

from __future__ import annotations

import os

from glc import providers
from glc.routing import Router


def _providers(monkeypatch, keys: dict[str, str]):
    for name in list(os.environ):
        if name.startswith("GEMINI_API_KEY"):
            monkeypatch.delenv(name, raising=False)
    for name, value in keys.items():
        monkeypatch.setenv(name, value)
    return providers.build_providers(cache_store=object())


def test_numbered_keys_are_real_providers_and_logical_gemini_expands(monkeypatch):
    pool = _providers(monkeypatch, {
        "GEMINI_API_KEY_1": "one",
        "GEMINI_API_KEY_2": "two",
        "GEMINI_API_KEY_3": "three",
    })
    assert {name for name in pool if name.startswith("gemini_")} == {"gemini_1", "gemini_2", "gemini_3"}
    assert [pool[f"gemini_{i}"].api_key for i in (1, 2, 3)] == ["one", "two", "three"]

    router = Router(pool, ["gemini"])
    assert router.candidates() == ["gemini_1", "gemini_2", "gemini_3"]

    first, _ = router.pick(100, router.candidates())
    assert first == "gemini_1"
    router.state[first].record(0)
    second, _ = router.pick(100, router.candidates())
    assert second == "gemini_2"
    router.state[second].mark_unavailable(60, "test quota")
    third, _ = router.pick(100, router.candidates())
    assert third == "gemini_3"


def test_legacy_single_key_is_a_one_member_pool(monkeypatch):
    pool = _providers(monkeypatch, {"GEMINI_API_KEY": "legacy"})
    assert {name for name in pool if name.startswith("gemini_")} == {"gemini_1"}
    assert Router(pool, ["gemini"]).candidates() == ["gemini_1"]

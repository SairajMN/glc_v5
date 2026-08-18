"""Server-side image fetches must not become a confused deputy.

v5 fetched caller-supplied image URLs with a bare httpx.get: no host
validation, follow_redirects=True with no per-hop check, no size cap and no
protection against DNS rebinding. The URL reaches this path from channel
adapters that accept inbound messages from the public internet.
"""

from __future__ import annotations

import ipaddress

import pytest
from fastapi import HTTPException

from glc.security import ssrf


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/admin",
        "http://localhost:8111/v1/control/kill",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.5/internal",
        "http://192.168.1.1/router",
        "http://172.16.0.1/",
        "http://[::1]/",
        "http://0.0.0.0/",
    ],
)
def test_internal_targets_are_rejected(url):
    with pytest.raises(HTTPException) as ei:
        ssrf.prepare_safe_target(url)
    assert ei.value.status_code == 400


@pytest.mark.parametrize("url", ["file:///etc/passwd", "gopher://x/", "ftp://x/", "data:text/plain,hi"])
def test_non_http_schemes_are_rejected(url):
    with pytest.raises(HTTPException) as ei:
        ssrf.prepare_safe_target(url)
    assert ei.value.status_code == 400


def test_ipv4_mapped_ipv6_is_rejected():
    """::ffff:169.254.169.254 must not smuggle the metadata address through."""
    assert ssrf._ip_is_forbidden(ipaddress.ip_address("::ffff:169.254.169.254"))
    assert ssrf._ip_is_forbidden(ipaddress.ip_address("::ffff:127.0.0.1"))


def test_sixtofour_wrapper_is_rejected():
    """6to4 (2002::/16) wrapping private space must be unwrapped and rejected."""
    assert ssrf._ip_is_forbidden(ipaddress.ip_address("2002:0a00:0001::"))  # 10.0.0.1


def test_public_address_is_allowed():
    assert not ssrf._ip_is_forbidden(ipaddress.ip_address("93.184.216.34"))
    assert not ssrf._ip_is_forbidden(ipaddress.ip_address("2606:2800:220:1:248:1893:25c8:1946"))


def test_hostname_resolving_to_private_ip_is_rejected(monkeypatch):
    """A public *name* whose A record is private must still be refused.

    This is the shape a literal-IP blocklist misses: the attacker controls
    DNS, not the URL text.
    """
    monkeypatch.setattr(
        ssrf, "_resolve_ips", lambda host: [ipaddress.ip_address("169.254.169.254")]
    )
    with pytest.raises(HTTPException) as ei:
        ssrf.prepare_safe_target("https://totally-innocent.example.com/cat.png")
    assert "disallowed" in str(ei.value.detail)


def test_safe_target_pins_the_resolved_ip(monkeypatch):
    """The hostname stays in the URL (SNI/Host) while connect uses the pin."""
    monkeypatch.setattr(ssrf, "_resolve_ips", lambda host: [ipaddress.ip_address("93.184.216.34")])
    target = ssrf.prepare_safe_target("https://example.com/cat.png")
    assert target.hostname == "example.com"
    assert target.pinned_ip == "93.184.216.34"
    assert target.url == "https://example.com/cat.png"


def test_redirects_are_not_followed_automatically():
    """Manual redirect handling is what lets every hop be re-validated."""
    import inspect

    src = inspect.getsource(ssrf.fetch_bytes)
    assert "follow_redirects=False" in src
    assert "prepare_safe_target(current)" in src


def test_size_cap_is_enforced_during_download_not_after():
    """A cap checked after buffering cannot prevent the allocation."""
    import inspect

    src = inspect.getsource(ssrf.fetch_bytes)
    assert "aiter_bytes" in src, "body must be streamed, not buffered whole"
    assert "content-length" in src, "an honestly declared oversized body should be refused early"


def test_chat_resolver_uses_the_guard():
    """The image path must call the guard rather than a bare httpx client."""
    import inspect

    from glc.routes import chat as chat_mod

    src = inspect.getsource(chat_mod._resolve_image_urls)
    assert "ssrf.fetch_bytes" in src
    assert "follow_redirects=True" not in src

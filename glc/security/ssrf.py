"""SSRF guards for server-side URL fetches.

The chat plane fetches caller-supplied image URLs and inlines them as data:
URLs before handing them to a provider. Without a guard a caller can point
those URLs at internal services (cloud metadata endpoints, loopback admin
ports, RFC1918 hosts) and use the gateway as a confused deputy. That matters
here even though GLC binds loopback by default, because channel adapters
accept inbound messages from the public internet and their content reaches
this same fetch path.

Redirects make validating the literal URL insufficient: a public host can
302 to http://169.254.169.254/. So redirects are followed manually and every
hop is re-validated.

Validating and then calling httpx.get(hostname) is a DNS-rebinding TOCTOU:
the check can see a public A record while the connect that follows resolves
to a private IP. Fetches therefore pin the validated IP at the transport
layer while keeping the hostname in the URL, so TLS SNI, certificate
verification and the HTTP Host header all still use the logical name.

The size cap is enforced *during* the download. A cap checked after the body
has been buffered cannot prevent the allocation it exists to prevent.
"""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urlparse

import httpcore
import httpx
from fastapi import HTTPException
from httpcore._backends.auto import AutoBackend

# Follow at most this many redirect hops (each re-validated).
MAX_REDIRECTS = 4
# Cap the fetched body so a huge internal blob cannot be amplified.
MAX_IMAGE_BYTES = 12 * 1024 * 1024  # 12 MiB
FETCH_TIMEOUT = 30


def _ip_is_forbidden(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """True if `ip` must never be reachable from a server-side fetch."""
    if (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    ):
        return True
    # IPv6 wrappers around forbidden IPv4 space (e.g. ::ffff:169.254.169.254,
    # 6to4 2002::/16). Unwrap and re-check the embedded IPv4.
    if isinstance(ip, ipaddress.IPv6Address):
        if getattr(ip, "is_site_local", False):
            return True
        mapped = ip.ipv4_mapped or ip.sixtofour
        if mapped is not None and _ip_is_forbidden(mapped):
            return True
    return False


def _resolve_ips(host: str) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    """Resolve `host` (name or literal) to every IP it maps to."""
    ips: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    # A bare IP literal must still be validated (never trust getaddrinfo alone).
    try:
        ips.append(ipaddress.ip_address(host))
        return ips
    except ValueError:
        pass
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        raise HTTPException(400, f"cannot resolve host {host!r}: {e}") from e
    for info in infos:
        raw = info[4][0]
        try:
            ips.append(ipaddress.ip_address(raw.split("%")[0]))
        except ValueError:
            continue
    if not ips:
        raise HTTPException(400, f"cannot resolve host {host!r}") from None
    return ips


def assert_safe_url(url: str) -> str:
    """Validate a single URL. Raises HTTPException(400) if unsafe."""
    prepare_safe_target(url)
    return url


@dataclass(frozen=True)
class SafeTarget:
    """A URL that passed SSRF checks, with the IP to dial for connect."""

    url: str
    hostname: str
    pinned_ip: str


def prepare_safe_target(url: str) -> SafeTarget:
    """Resolve DNS once, reject forbidden addresses, return the connect pin.

    The returned ``url`` keeps the original hostname so httpx uses it for SNI,
    certificate verification and the Host header. ``pinned_ip`` is what the
    transport must dial, which is what closes the rebinding window.
    """
    if not isinstance(url, str):
        raise HTTPException(400, "url must be a string") from None
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(
            400, f"unsupported url scheme {parsed.scheme!r}; only http/https allowed"
        ) from None
    host = parsed.hostname
    if not host:
        raise HTTPException(400, "url has no host") from None
    ips = _resolve_ips(host)
    for ip in ips:
        if _ip_is_forbidden(ip):
            raise HTTPException(
                400, "url resolves to a disallowed (private/loopback/reserved) address"
            ) from None
    return SafeTarget(url=url, hostname=host, pinned_ip=str(ips[0]))


class _PinningBackend(AutoBackend):
    """Dial a pinned IP while httpx still names the logical host (SNI/Host)."""

    def __init__(self, pins: dict[str, str]) -> None:
        super().__init__()
        self._pins = {h.lower().rstrip("."): ip for h, ip in pins.items()}

    async def connect_tcp(  # type: ignore[override]
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options=None,
    ) -> httpcore.AsyncNetworkStream:
        key = host.decode() if isinstance(host, (bytes, bytearray)) else str(host)
        dial = self._pins.get(key.lower().rstrip("."), key)
        return await super().connect_tcp(
            dial,
            port,
            timeout=timeout,
            local_address=local_address,
            socket_options=socket_options,
        )


def _pinned_transport(pins: dict[str, str]) -> httpx.AsyncHTTPTransport:
    """httpx transport whose TCP connect uses pinned IPs (SNI stays on URL host)."""
    from httpx._config import DEFAULT_LIMITS, create_ssl_context

    transport = httpx.AsyncHTTPTransport()
    transport._pool = httpcore.AsyncConnectionPool(
        ssl_context=create_ssl_context(verify=True, cert=None, trust_env=True),
        max_connections=DEFAULT_LIMITS.max_connections,
        max_keepalive_connections=DEFAULT_LIMITS.max_keepalive_connections,
        keepalive_expiry=DEFAULT_LIMITS.keepalive_expiry,
        http1=True,
        http2=False,
        network_backend=_PinningBackend(pins),
    )
    return transport


async def fetch_bytes(url: str) -> tuple[bytes, str]:
    """Fetch a URL safely: validate and IP-pin every hop, no automatic redirects.

    Returns (content, content_type). Raises HTTPException(400) on any
    validation failure, size violation or transport error.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; GLCv1/0.1; +image-resolver)",
        "Accept": "image/*,*/*;q=0.8",
    }
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        target = prepare_safe_target(current)
        transport = _pinned_transport({target.hostname: target.pinned_ip})
        try:
            async with httpx.AsyncClient(
                timeout=FETCH_TIMEOUT,
                follow_redirects=False,
                headers=headers,
                transport=transport,
            ) as client:
                # Keep the hostname in the URL so TLS SNI and cert verification
                # use it; the transport dials target.pinned_ip instead.
                async with client.stream("GET", target.url) as r:
                    if r.is_redirect:
                        location = r.headers.get("location")
                        if not location:
                            raise HTTPException(400, "redirect without location header")
                        current = str(httpx.URL(current).join(location))
                        continue
                    r.raise_for_status()
                    # An honestly declared oversized body is refused before a
                    # single chunk is read.
                    declared = r.headers.get("content-length")
                    if declared and declared.isdigit() and int(declared) > MAX_IMAGE_BYTES:
                        raise HTTPException(400, "fetched resource exceeds size cap")
                    chunks: list[bytes] = []
                    total = 0
                    async for chunk in r.aiter_bytes():
                        total += len(chunk)
                        # Enforced during the download: a cap checked after the
                        # body is buffered cannot prevent the allocation.
                        if total > MAX_IMAGE_BYTES:
                            raise HTTPException(400, "fetched resource exceeds size cap")
                        chunks.append(chunk)
                    ctype = (r.headers.get("content-type") or "image/png").split(";")[0].strip()
                    return b"".join(chunks), ctype
        except httpx.HTTPError as e:
            raise HTTPException(400, f"failed to fetch url: {e}") from e
    raise HTTPException(400, "too many redirects") from None

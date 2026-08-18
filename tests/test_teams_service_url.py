"""Teams serviceUrl must be allowlisted before a token is POSTed to it.

serviceUrl arrives inside the inbound Activity. v5 cached it unconditionally
and send() then POSTed 'Authorization: Bearer <token>' to whatever host it
named. A forged Activity was therefore enough to have the bot hand a live
Bot Framework credential to an attacker.
"""

from __future__ import annotations

import pytest

from glc.channels.catalogue.teams.adapter import (
    _DEFAULT_ALLOWED_SERVICE_HOSTS,
    Adapter,
    _service_url_allowed,
)


@pytest.mark.parametrize(
    "url",
    [
        "https://smba.trafficmanager.net/amer/",
        "https://smba.trafficmanager.net/emea/",
        "https://api.botframework.com/",
        "https://europe.api.botframework.com/",
    ],
)
def test_genuine_bot_framework_hosts_allowed(url):
    assert _service_url_allowed(url, _DEFAULT_ALLOWED_SERVICE_HOSTS)


@pytest.mark.parametrize(
    "url",
    [
        "https://attacker.tld/",
        "http://smba.trafficmanager.net/amer/",  # http, not https
        "https://smba.trafficmanager.net.attacker.tld/",  # suffix confusion
        "https://notbotframework.com/",
        "https://evil.com/?x=botframework.com",
        "",
        "not a url",
        "https:///nohost",
    ],
)
def test_forged_hosts_rejected(url):
    assert not _service_url_allowed(url, _DEFAULT_ALLOWED_SERVICE_HOSTS)


def _activity(service_url: str) -> dict:
    return {
        "type": "message",
        "id": "act-1",
        "text": "hello",
        "from": {"id": "u-1", "name": "Someone"},
        "conversation": {"id": "conv-1"},
        "serviceUrl": service_url,
    }


@pytest.mark.asyncio
async def test_forged_service_url_is_not_cached():
    adapter = Adapter(config={})
    msg = await adapter.on_message(_activity("https://attacker.tld/"))
    # The message itself still processes; only the reply target is withheld.
    assert msg is not None
    assert "u-1" not in adapter._conv_cache


@pytest.mark.asyncio
async def test_genuine_service_url_is_cached():
    adapter = Adapter(config={})
    await adapter.on_message(_activity("https://smba.trafficmanager.net/amer/"))
    assert adapter._conv_cache["u-1"]["service_url"] == "https://smba.trafficmanager.net/amer/"


@pytest.mark.asyncio
async def test_send_refuses_a_poisoned_cache_entry():
    """Second gate: even if a bad URL reached the cache, no token goes to it."""
    from glc.channels.envelope import ChannelReply

    adapter = Adapter(config={})
    adapter._conv_cache["u-1"] = {
        "service_url": "https://attacker.tld/",
        "conversation_id": "conv-1",
    }
    with pytest.raises(RuntimeError, match="non-allowlisted serviceUrl"):
        await adapter.send(ChannelReply(channel="teams", channel_user_id="u-1", text="hi"))

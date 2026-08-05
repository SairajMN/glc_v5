"""WS /v1/channels/{name} — adapter control plane.

Adapters connect over WebSocket and exchange JSON-serialised
ChannelMessage and ChannelReply envelopes. The connection is gated by
the installation token presented in the Authorization header (Sec-Websocket
clients can pass it as a query string fallback, ?token=...).

This endpoint is the contract surface adapters speak to. The gateway
processes incoming messages through the rate limiter, allowlist and audit
boundary, then forwards the shared envelope to S16Code. No channel-specific
agent integration lives here: every discovered adapter uses the same bridge.
"""

from __future__ import annotations

import hmac
import json
import os

from fastapi import APIRouter, Header, HTTPException, Query, Request, WebSocket, WebSocketDisconnect, status
from fastapi.responses import JSONResponse, PlainTextResponse

from glc.audit import append as audit_append
from glc.channels import registry
from glc.channels.agent_bridge import AgentBridgeError
from glc.channels.envelope import ChannelMessage, ChannelReply
from glc.config import get_or_create_install_token
from glc.security.allowlists import allowed
from glc.security.pairing import get_pairing_store
from glc.security.rate_limits import get_rate_limiter

router = APIRouter()


def _bridge(state):
    bridge = getattr(state, "agent_bridge", None)
    if bridge is None:
        raise AgentBridgeError("S16 agent bridge is not configured")
    return bridge


def _authorised_channel_control(authorization: str | None) -> bool:
    presented = (authorization or "").removeprefix("Bearer ").strip()
    candidates = {get_or_create_install_token()}
    bridge_token = os.getenv("GLC_S16_BRIDGE_TOKEN", "").strip()
    if bridge_token:
        candidates.add(bridge_token)
    return bool(presented) and any(hmac.compare_digest(presented, expected) for expected in candidates)


def _verified_identity(message: ChannelMessage, pairings) -> ChannelMessage:
    """Replace adapter/client identity claims with gateway-owned pairing state."""
    record = pairings.lookup(message.channel, message.channel_user_id)
    trust = record.trust_level if record is not None else "untrusted"
    principal = ("installation-owner" if trust == "owner_paired"
                 else f"{message.channel}:{message.channel_user_id}")
    metadata = {**message.metadata, "glc_principal_id": principal}
    return message.model_copy(update={"trust_level": trust, "metadata": metadata})


@router.websocket("/v1/channels/{name}")
async def channel_ws(websocket: WebSocket, name: str, token: str | None = Query(default=None)):
    header_auth = websocket.headers.get("authorization") or websocket.headers.get("Authorization")
    presented = None
    if header_auth and header_auth.startswith("Bearer "):
        presented = header_auth.removeprefix("Bearer ").strip()
    elif token:
        presented = token
    expected = get_or_create_install_token()
    if presented != expected:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await websocket.accept()
    state = websocket.app.state
    registered = list(getattr(state, "registered_channels", []))
    if name not in registered:
        registered.append(name)
        state.registered_channels = registered

    limiter = get_rate_limiter()
    pairings = get_pairing_store()
    owners = [p.channel_user_id for p in pairings.owners(channel=name)]

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                payload = json.loads(raw)
                env = ChannelMessage.model_validate(payload)
            except Exception as e:
                await websocket.send_text(json.dumps({"error": f"invalid envelope: {e}"}))
                continue

            if env.channel != name:
                await websocket.send_text(json.dumps({"error": "envelope channel does not match route"}))
                continue
            env = _verified_identity(env, pairings)

            ok, why = allowed(
                env.channel,
                env.channel_user_id,
                owner_ids=owners,
                is_public_channel=bool(env.metadata.get("is_public_channel", False)),
                was_mentioned=bool(env.metadata.get("was_mentioned", False)),
            )
            if not ok:
                audit_append(
                    channel=env.channel,
                    channel_user_id=env.channel_user_id,
                    trust_level=env.trust_level,
                    event_type="allowlist_drop",
                    result={"reason": why},
                )
                await websocket.send_text(json.dumps({"error": f"dropped: {why}"}))
                continue

            ok, why = limiter.check_message(env.channel, env.channel_user_id)
            if not ok:
                audit_append(
                    channel=env.channel,
                    channel_user_id=env.channel_user_id,
                    trust_level=env.trust_level,
                    event_type="rate_limit",
                    result={"reason": why},
                )
                await websocket.send_text(json.dumps({"status": 429, "error": why}))
                continue

            audit_append(
                channel=env.channel,
                channel_user_id=env.channel_user_id,
                trust_level=env.trust_level,
                event_type="inbound_message",
                params={"text": env.text, "thread_id": env.thread_id},
            )

            try:
                reply = await _bridge(state).handle(env)
            except AgentBridgeError as error:
                audit_append(
                    channel=env.channel,
                    channel_user_id=env.channel_user_id,
                    trust_level=env.trust_level,
                    event_type="agent_bridge_error",
                    result={"error": str(error)},
                )
                await websocket.send_text(json.dumps({"status": 503, "error": str(error)}))
                continue
            await websocket.send_text(reply.model_dump_json())
    except WebSocketDisconnect:
        return


@router.get("/v1/channels/{name}/webhook")
async def channel_webhook_verify(name: str, request: Request):
    params = dict(request.query_params)
    mode = params.get("hub.mode", "")
    token = params.get("hub.verify_token", "")
    challenge = params.get("hub.challenge", "")
    expected = os.environ.get(f"{name.upper()}_VERIFY_TOKEN", "")
    if mode == "subscribe" and hmac.compare_digest(token, expected):
        return PlainTextResponse(challenge)
    raise HTTPException(status_code=403)


@router.post("/v1/channels/{name}/webhook")
async def channel_webhook(name: str, request: Request):
    try:
        adapter = registry.instantiate(name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown channel: {name}") from None

    raw = {
        "raw_body": await request.body(),
        "headers": dict(request.headers),
    }
    msg = await adapter.on_message(raw)
    if msg is None:
        return {"status": "ok"}

    limiter = get_rate_limiter()
    pairings = get_pairing_store()
    msg = _verified_identity(msg, pairings)
    owners = [p.channel_user_id for p in pairings.owners(channel=name)]

    ok, why = allowed(
        msg.channel,
        msg.channel_user_id,
        owner_ids=owners,
        is_public_channel=bool(msg.metadata.get("is_public_channel", False)),
        was_mentioned=bool(msg.metadata.get("was_mentioned", False)),
    )
    if not ok:
        audit_append(
            channel=msg.channel,
            channel_user_id=msg.channel_user_id,
            trust_level=msg.trust_level,
            event_type="allowlist_drop",
            result={"reason": why},
        )
        return {"status": "ok"}

    ok, why = limiter.check_message(msg.channel, msg.channel_user_id)
    if not ok:
        audit_append(
            channel=msg.channel,
            channel_user_id=msg.channel_user_id,
            trust_level=msg.trust_level,
            event_type="rate_limit",
            result={"reason": why},
        )
        return JSONResponse(status_code=429, content={"error": why})

    audit_append(
        channel=msg.channel,
        channel_user_id=msg.channel_user_id,
        trust_level=msg.trust_level,
        event_type="inbound_message",
        params={"text": msg.text, "thread_id": msg.thread_id, "provider": msg.metadata.get("provider")},
    )

    try:
        reply = await _bridge(request.app.state).handle(msg)
    except AgentBridgeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    await adapter.send(reply)
    return {"status": "ok"}


@router.get("/v1/channels")
async def channel_catalogue(request: Request):
    """The dynamic channel surface S16 can inspect without a hard-coded list."""
    connected = set(getattr(request.app.state, "registered_channels", []))
    return {
        "channels": [
            {"name": name, "connected": name in connected}
            for name in registry.list_channels()
        ]
    }


@router.post("/v1/channels/{name}/send")
async def send_channel_message(
    name: str,
    reply: ChannelReply,
    request: Request,
    authorization: str | None = Header(default=None),
):
    """Dispatch one proactive message through any registered adapter.

    This is intentionally one generic capability. Provider-specific addressing
    remains in the adapter and the model cannot invent an unregistered channel.
    """
    if not _authorised_channel_control(authorization):
        raise HTTPException(status_code=401, detail="invalid channel bridge token")
    if reply.channel != name:
        raise HTTPException(status_code=422, detail="body channel does not match route")
    try:
        adapter = registry.instantiate(name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown channel: {name}") from None
    try:
        result = await adapter.send(reply)
    except Exception as error:
        audit_append(
            channel=name,
            channel_user_id=reply.channel_user_id,
            trust_level="owner_paired",
            event_type="outbound_message_failed",
            params={"thread_id": reply.thread_id},
            result={"error": f"{type(error).__name__}: {error}"},
        )
        raise HTTPException(status_code=502, detail=f"channel send failed: {type(error).__name__}: {error}") from error
    audit_append(
        channel=name,
        channel_user_id=reply.channel_user_id,
        trust_level="owner_paired",
        event_type="outbound_message",
        params={"thread_id": reply.thread_id},
        result={"adapter_result": result},
    )
    return {"accepted": True, "channel": name, "adapter_result": result}

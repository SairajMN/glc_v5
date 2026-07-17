# GLC v1 architecture

Session 11 §7 names six architectural moves that distinguish GLC v1
from OpenClaw's default posture. Each is a direct response to a
documented OpenClaw failure mode. The mapping below names the move,
where it lives in this repository, and which incident it answers.

## 1. The agent does not get host execution authority by default

The runtime is the S9 multi-agent DAG plus the typed skill catalogue.
Tools live as named typed handlers; there is no generic `shell.exec`
endpoint that the agent can reach by default. Channel adapters speak
typed Pydantic envelopes (`ChannelMessage`, `ChannelReply`) — they
never hand a Discord embed or a Telegram InputFile to the agent.

Lives in: `glc/channels/envelope.py`, `glc/channels/base.py`.
Answers: ClawJacked.

## 2. The policy engine runs outside the LLM context

`glc/policy/engine.py` is a small declarative evaluator. Rules are
specified in `glc/policy/policy.yaml` (or `~/.glc/policy.yaml` to
override). Every tool call is intercepted before dispatch and evaluated.
First matching rule wins; ties resolve to deny. The defaults for
`owner_paired` and `untrusted` are allow and deny respectively. The
yaml hot-reloads on `SIGHUP`.

Lives in: `glc/policy/`.
Answers: the Summer Yue email-deletion incident, where the "confirm
before acting" rule lived inside the conversation context and was
erased by context compaction. Yaml does not compact.

## 3. Memory is classed with per-class write permissions

The audit store (`glc/audit/`) is append-only: only `append()` is
exposed; no `update`, no `delete`. The pairing store, the channels
configuration, and the policy rules all live outside the LLM's
write reach. Sessions 12 and onward expand the memory taxonomy to
working / episodic / semantic / procedural classes; the
write-permission discipline begins here.

Lives in: `glc/audit/`, `glc/security/pairing.py`, `glc/config.py`.
Answers: persistent-prompt-injection attacks that would otherwise
mutate the agent's own constraints.

## 4. The control plane has an out-of-band path

`/v1/control/kill`, `/v1/control/pair`, `/v1/control/presence` are
authenticated by a per-installation token (`~/.glc/install_token`)
and isolated from the channel traffic. The kill endpoint binds
127.0.0.1 by default; bypassing this requires `GLC_KILL_ALLOW_REMOTE=1`
and a deliberate operator decision. The intent is to make "STOP"
reachable through a phone-friendly URL the user can hit when the
agent has gone off the rails inside the channel.

Lives in: `glc/routes/control.py`.
Answers: Summer Yue having to physically reach the Mac mini.

## 5. Every channel envelope carries a trust level

`TrustLevel` is `owner_paired | user_paired | untrusted`. Adapters
classify inbound messages via `glc/security/trust_level.py`, which
consults the pairing store. The policy engine reads the trust level
before authorising any tool action; the lecture's default rule
denies all tools for `untrusted`.

Lives in: `glc/channels/envelope.py`, `glc/security/trust_level.py`,
`glc/policy/`.
Answers: confused-deputy and indirect-prompt-injection. An email
scraped from the inbox can carry instructions, but it arrives with
`trust_level=untrusted` and the policy engine rejects everything.

## 6. Every action is logged append-only

`glc/audit/store.py` writes to `~/.glc/audit.sqlite`. Each row carries
session id, channel, sender id, trust level, event type, tool, policy
verdict, params, result. The S8 replay viewer reads from the same
store. Each insert commits immediately so the trail survives a hard
kill.

Lives in: `glc/audit/`.
Answers: the recovery scenario after a bad outcome — the operator
can replay exactly what the agent saw and did.

## What S12 works on

The S11 policy engine is the application-level enforcement layer.
S12 moves this gateway onto Modal containers and then hardens it. The
environmental layer it needs, container isolation per component,
short-lived scoped credentials in place of one shared secret, and
network egress filters, is not built yet. Building it is the
assignment. The two layers should run independently, so that a
failure in one does not invalidate the other.

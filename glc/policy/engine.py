"""Declarative policy engine.

evaluate(tool_call, context) -> PolicyVerdict
  - first matching rule wins
  - ties resolve to deny
  - default allow when trust_level == 'owner_paired' and no rule matches
  - default deny otherwise

Hot reload on SIGHUP (process-level handler installed by main.py). Malformed
yaml is rejected: the engine falls back to a deny-everything safe-default
config and logs a warning so the gateway boots in a known-safe state.
"""

from __future__ import annotations

import fnmatch
import os
import re
import threading
from pathlib import Path
from typing import Any

import yaml

from glc.policy.schemas import PolicyConfig, PolicyRule, PolicyVerdict

_SAFE_DEFAULT = PolicyConfig(
    rules=[
        PolicyRule(
            tool="*",
            trust_level="*",
            action="deny",
            reason="policy.yaml unreadable — falling back to deny-everything",
        )
    ]
)


def _normalize_path_for_glob(s: str) -> str:
    """Expand ``~`` and normalize separators so a rule written as
    ``~/Documents/**`` also matches the equivalent absolute path on the host
    (e.g. ``/Users/x/Documents/...``). Without this, spelling the same file
    absolutely slips past a tilde-based deny rule."""
    return os.path.expanduser(s).replace("\\", "/")


def _matches_glob(value: Any, pattern: str) -> bool:
    # A non-string value where a string is expected is type-confusion. Raise
    # so evaluate() can fail CLOSED — silently returning False here skips the
    # deny rule and falls through to default-allow.
    if not isinstance(value, str):
        raise TypeError(f"glob condition expected str, got {type(value).__name__}")
    if not isinstance(pattern, str):
        raise TypeError(f"glob pattern expected str, got {type(pattern).__name__}")
    value = _normalize_path_for_glob(value)
    pattern = _normalize_path_for_glob(pattern)
    # fnmatch's ** support is weak; substitute ** for a regex-ish pattern.
    if "**" in pattern:
        regex = re.escape(pattern).replace(r"\*\*", ".*").replace(r"\*", "[^/]*")
        # fullmatch + DOTALL anchors the WHOLE string and lets ``.*`` cross
        # newlines. re.match with a trailing "$" stops at the first newline,
        # so a `\n` injected into the value evades the deny rule.
        return bool(re.fullmatch(regex, value, re.DOTALL))
    # fnmatch.translate anchors with \Z and enables DOTALL, so it is already
    # newline-safe.
    return fnmatch.fnmatch(value, pattern)


def _matches_regex(value: Any, pattern: str) -> bool:
    # Fail CLOSED on non-string, as in _matches_glob.
    if not isinstance(value, str):
        raise TypeError(f"regex condition expected str, got {type(value).__name__}")
    # DOTALL so a `\n` in the value cannot hide content from the pattern.
    return bool(re.search(pattern, value, re.DOTALL))


def _command_matches(command: Any, patterns: list[Any]) -> bool:
    # Fail CLOSED on non-string command.
    if not isinstance(command, str):
        raise TypeError(f"command_matches expected str, got {type(command).__name__}")
    # Casefold both sides so `SUDO` cannot bypass a lowercase deny rule.
    haystack = command.casefold()
    return any(str(p).casefold() in haystack for p in patterns)


def _matches_condition(condition: dict[str, Any], params: dict[str, Any]) -> bool:
    for key, expected in condition.items():
        if key.endswith("_glob"):
            target = key[: -len("_glob")]
            if not _matches_glob(params.get(target), expected):
                return False
        elif key.endswith("_regex"):
            target = key[: -len("_regex")]
            if not _matches_regex(params.get(target), expected):
                return False
        elif key.endswith("_in"):
            target = key[: -len("_in")]
            if params.get(target) not in (expected or []):
                return False
        elif key == "command_matches":
            patterns = expected if isinstance(expected, list) else [expected]
            if not _command_matches(params.get("command", ""), patterns):
                return False
        elif key == "recipient_type":
            if params.get("recipient_type") != expected:
                return False
        elif isinstance(expected, list):
            if params.get(key) not in expected:
                return False
        else:
            if params.get(key) != expected:
                return False
    return True


class PolicyEngine:
    def __init__(self, config: PolicyConfig):
        self.config = config
        self._lock = threading.Lock()

    @classmethod
    def from_yaml(cls, path: Path | str) -> PolicyEngine:
        p = Path(path)
        if not p.exists():
            return cls(_SAFE_DEFAULT)
        try:
            raw = yaml.safe_load(p.read_text()) or {}
            cfg = PolicyConfig.model_validate(raw)
        except Exception as e:  # pragma: no cover
            print(f"[glc.policy] malformed {p}: {e!r} — using deny-everything safe default")
            cfg = _SAFE_DEFAULT
        return cls(cfg)

    def evaluate(self, tool_call: dict[str, Any], context: dict[str, Any]) -> PolicyVerdict:
        """tool_call = {'name': 'email.send', 'arguments': {...}}
        context   = {'channel': 'telegram', 'trust_level': 'owner_paired',
                     'channel_user_id': '...'}"""
        tool = tool_call.get("name", "")
        params = tool_call.get("arguments") or {}
        channel = context.get("channel", "")
        trust_level = context.get("trust_level", "untrusted")

        with self._lock:
            rules = list(self.config.rules)

        deny_match: tuple[int, PolicyRule] | None = None
        first_match: tuple[int, PolicyRule] | None = None
        for i, rule in enumerate(rules):
            if rule.tool != "*" and rule.tool != tool:
                continue
            if rule.channel != "*" and rule.channel != channel:
                continue
            if rule.trust_level != "*" and rule.trust_level != trust_level:
                continue
            try:
                condition_ok = not rule.condition or _matches_condition(rule.condition, params)
            except TypeError:
                # A matcher got a non-string argument where a string was
                # required (glob / regex / command). Fail CLOSED: treat the
                # condition as satisfied so a deny rule still fires, rather
                # than skipping it and falling through to default-allow.
                condition_ok = True
            if not condition_ok:
                continue
            if first_match is None:
                first_match = (i, rule)
            if rule.action == "deny" and deny_match is None:
                deny_match = (i, rule)

        if deny_match is not None:
            i, r = deny_match
            return PolicyVerdict(action="deny", reason=r.reason or "denied by policy", matched_rule_index=i)
        if first_match is not None:
            i, r = first_match
            return PolicyVerdict(
                action=r.action, reason=r.reason or f"matched rule #{i}", matched_rule_index=i
            )
        if trust_level == "owner_paired":
            return PolicyVerdict(action="allow", reason="default-allow for owner_paired")
        return PolicyVerdict(action="deny", reason=f"default-deny for trust_level={trust_level}")

    def reload(self, path: Path | str) -> None:
        new = PolicyEngine.from_yaml(path)
        with self._lock:
            self.config = new.config


# Module-level singleton, lazily constructed from config.policy_yaml_path().
_engine: PolicyEngine | None = None
_engine_lock = threading.Lock()


def get_engine() -> PolicyEngine:
    global _engine
    with _engine_lock:
        if _engine is None:
            from glc.config import policy_yaml_path

            _engine = PolicyEngine.from_yaml(policy_yaml_path())
    return _engine


def reload_engine() -> None:
    from glc.config import policy_yaml_path

    eng = get_engine()
    eng.reload(policy_yaml_path())


def evaluate(tool_call: dict[str, Any], context: dict[str, Any]) -> PolicyVerdict:
    return get_engine().evaluate(tool_call, context)

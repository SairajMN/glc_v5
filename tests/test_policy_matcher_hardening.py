"""Policy matcher bypasses backported from glc_v2.

Each test here is a way to get a deny rule to not fire. They share one shape:
the matcher is handed something it does not expect, quietly reports "no
match", the rule is skipped, and evaluation falls through to default-allow
for an owner_paired caller. The engine must fail closed instead.
"""

from __future__ import annotations

import os

import pytest

from glc.policy.engine import PolicyEngine
from glc.policy.schemas import PolicyConfig, PolicyRule

OWNER = {"channel": "telegram", "trust_level": "owner_paired"}


def _engine(rule: PolicyRule) -> PolicyEngine:
    eng = PolicyEngine.__new__(PolicyEngine)
    import threading

    eng._lock = threading.Lock()
    eng.config = PolicyConfig(rules=[rule])
    return eng


def test_nonstring_glob_arg_does_not_skip_deny():
    """A dict where a path string was expected must not skip the deny rule."""
    eng = _engine(
        PolicyRule(
            tool="file.delete",
            trust_level="*",
            action="deny",
            reason="no deleting Documents",
            condition={"path_glob": "~/Documents/**"},
        )
    )
    v = eng.evaluate({"name": "file.delete", "arguments": {"path": {"$ne": None}}}, OWNER)
    assert v.action == "deny", v


def test_nonstring_command_does_not_skip_deny():
    eng = _engine(
        PolicyRule(
            tool="shell.run",
            trust_level="*",
            action="deny",
            reason="no sudo",
            condition={"command_matches": ["sudo"]},
        )
    )
    v = eng.evaluate({"name": "shell.run", "arguments": {"command": ["sudo", "rm"]}}, OWNER)
    assert v.action == "deny", v


def test_command_match_is_case_insensitive():
    """`SUDO` must not walk past a lowercase deny rule."""
    eng = _engine(
        PolicyRule(
            tool="shell.run",
            trust_level="*",
            action="deny",
            reason="no sudo",
            condition={"command_matches": ["sudo"]},
        )
    )
    v = eng.evaluate({"name": "shell.run", "arguments": {"command": "SUDO rm -rf /"}}, OWNER)
    assert v.action == "deny", v


def test_absolute_path_matches_tilde_rule():
    """The same file spelled absolutely must hit a ``~/`` deny rule."""
    eng = _engine(
        PolicyRule(
            tool="file.delete",
            trust_level="*",
            action="deny",
            reason="no deleting Documents",
            condition={"path_glob": "~/Documents/**"},
        )
    )
    absolute = os.path.expanduser("~/Documents/taxes.pdf")
    v = eng.evaluate({"name": "file.delete", "arguments": {"path": absolute}}, OWNER)
    assert v.action == "deny", v


def test_newline_in_value_does_not_evade_glob():
    """A trailing newline must not truncate the match and evade the rule."""
    eng = _engine(
        PolicyRule(
            tool="file.delete",
            trust_level="*",
            action="deny",
            reason="no deleting Documents",
            condition={"path_glob": "~/Documents/**"},
        )
    )
    sneaky = os.path.expanduser("~/Documents/taxes.pdf") + "\nnot-a-real-suffix"
    v = eng.evaluate({"name": "file.delete", "arguments": {"path": sneaky}}, OWNER)
    assert v.action == "deny", v


def test_ordinary_allow_still_works():
    """Failing closed must not turn unrelated allowed calls into denials."""
    eng = _engine(
        PolicyRule(
            tool="file.delete",
            trust_level="*",
            action="deny",
            reason="no deleting Documents",
            condition={"path_glob": "~/Documents/**"},
        )
    )
    v = eng.evaluate({"name": "file.read", "arguments": {"path": "/tmp/ok.txt"}}, OWNER)
    assert v.action == "allow", v


@pytest.mark.parametrize("bad", [123, None, {"a": 1}, ["x"]])
def test_nonstring_regex_arg_does_not_skip_deny(bad):
    eng = _engine(
        PolicyRule(
            tool="email.send",
            trust_level="*",
            action="deny",
            reason="no external recipients",
            condition={"to_regex": r".*@external\.com"},
        )
    )
    v = eng.evaluate({"name": "email.send", "arguments": {"to": bad}}, OWNER)
    assert v.action == "deny", v

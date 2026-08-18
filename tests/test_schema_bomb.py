"""response_format.schema must be bounded before validation (backport glc_v2 #25).

The schema arrives from the caller and went straight into
Draft202012Validator. A root-pointing $ref or a deeply nested structure makes
the validator recurse until the interpreter dies, which is a denial of
service on an authenticated data-plane call.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from glc.routes.chat import MAX_SCHEMA_DEPTH, _validate_structured, assert_schema_sane


def test_root_ref_is_rejected():
    with pytest.raises(HTTPException, match="self-referential") as ei:
        _validate_structured('{"x": 1}', {"$ref": "#"})
    assert ei.value.status_code == 400


def test_nested_root_ref_is_rejected():
    """Small enough to clear the depth and node caps, so it pins the $ref check."""
    with pytest.raises(HTTPException, match="self-referential") as ei:
        _validate_structured('{"a": 1}', {"properties": {"a": {"$ref": "#"}}})
    assert ei.value.status_code == 400


def test_deeply_nested_schema_is_rejected():
    deep: dict = {"type": "string"}
    for _ in range(MAX_SCHEMA_DEPTH + 5):
        deep = {"properties": {"x": deep}}
    with pytest.raises(HTTPException, match="too deeply nested") as ei:
        _validate_structured("{}", deep)
    assert ei.value.status_code == 400


def test_broad_schema_is_rejected():
    wide = {"properties": {f"k{i}": {"type": "string"} for i in range(MAX_SCHEMA_DEPTH * 400)}}
    with pytest.raises(HTTPException, match="too large") as ei:
        _validate_structured("{}", wide)
    assert ei.value.status_code == 400


def test_ordinary_defs_ref_still_resolves():
    """Only root-pointing refs are bombs; normal $defs use must keep working."""
    assert_schema_sane(
        {
            "type": "object",
            "properties": {"n": {"$ref": "#/$defs/pos"}},
            "$defs": {"pos": {"type": "integer", "minimum": 0}},
        }
    )


def test_valid_schema_still_validates():
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }
    assert _validate_structured('{"name": "Alice"}', schema) == {"name": "Alice"}


def test_schema_mismatch_still_raises():
    from jsonschema import ValidationError

    schema = {"type": "object", "properties": {"count": {"type": "integer"}}, "required": ["count"]}
    with pytest.raises(ValidationError):
        _validate_structured('{"count": "not-an-int"}', schema)

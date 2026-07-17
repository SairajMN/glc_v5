## Gateway change

What gateway behavior changes, and which caller depends on it?

## Reproduction and result

Provide the smallest reproduction from a fresh checkout. Show the result before and after the change.

## Boundary checklist

- [ ] Provider credentials remain inside `glc_v3`.
- [ ] No agent graph, memory, semantic-indexing, or A2A runtime was added here.
- [ ] Existing `/v1/*` callers remain compatible.
- [ ] No `.env`, credentials, local databases, audit records, or pairing state are committed.
- [ ] `uv run ruff check .` passes.
- [ ] `uv run pytest -q` passes.

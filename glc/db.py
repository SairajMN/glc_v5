"""V9-compatible per-call ledger. Same schema as llm_gatewayV9/db.py, but
the database lives under ~/.glc/ so the gateway is installable as a daemon
without writing into the source tree.

Note: this is the *worker call* ledger, used by /v1/cost/by_agent. The
audit log (every channel message, policy verdict, tool dispatch) is a
separate append-only store under glc/audit/store.py.
"""

from __future__ import annotations

import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

DEFAULT_DIR = Path(os.path.expanduser("~/.glc"))
DB_PATH = os.getenv("GLC_GATEWAY_DB", str(DEFAULT_DIR / "gateway.sqlite"))


def _ensure_parent() -> None:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def conn():
    _ensure_parent()
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    try:
        yield c
        c.commit()
    finally:
        c.close()


# ── v4 additive ledger migration ────────────────────────────────────────────
# The v3 `calls` table is keyed by agent+session. v4 adds the principal
# dimensions above those two, plus the money and cache columns that the
# economics layer needs. Every one is nullable with a benign default, and the
# migration is a plain ADD COLUMN, so:
#   * rows written by glc_v3 stay readable and keep their meaning,
#   * every v3 query (by_agent, recent, aggregate) is unchanged,
#   * a v4 gateway can be pointed at an existing ~/.glc/gateway.sqlite.
# Ordered dict: name -> column definition.
V4_COLUMNS: dict[str, str] = {
    "tenant": "TEXT",
    "project": "TEXT",
    "user": "TEXT",
    "usd": "REAL DEFAULT 0",
    "price_source": "TEXT",
    "cache_hit": "INTEGER DEFAULT 0",
    "cache_kind": "TEXT",
    "role": "TEXT",
    "tier": "TEXT",
    "escalations": "INTEGER DEFAULT 0",
    "tokens_saved": "INTEGER DEFAULT 0",
    "usd_saved": "REAL DEFAULT 0",
}

#: Attribution dimensions, coarsest first. These are ledger *columns*, so the
#: list is structural; which values appear in them is entirely caller-supplied.
PRINCIPAL_DIMENSIONS: tuple[str, ...] = ("tenant", "project", "user", "agent", "session")


def _existing_columns(c, table: str = "calls") -> set[str]:
    return {r["name"] for r in c.execute(f"PRAGMA table_info({table})").fetchall()}


def migrate() -> list[str]:
    """Bring an existing ledger up to the v4 shape. Returns columns added."""
    added: list[str] = []
    with conn() as c:
        have = _existing_columns(c)
        if not have:
            return added  # table does not exist yet; init() creates it complete
        for name, decl in V4_COLUMNS.items():
            if name not in have:
                c.execute(f'ALTER TABLE calls ADD COLUMN "{name}" {decl}')
                added.append(name)
        for dim in ("tenant", "project", "user"):
            c.execute(f'CREATE INDEX IF NOT EXISTS idx_{dim}_ts ON calls("{dim}", ts DESC)')
    return added


def init() -> None:
    with conn() as c:
        c.execute(
            """CREATE TABLE IF NOT EXISTS calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                cache_create_tokens INTEGER DEFAULT 0,
                cache_read_tokens INTEGER DEFAULT 0,
                latency_ms INTEGER DEFAULT 0,
                status TEXT,
                error TEXT,
                prompt_chars INTEGER DEFAULT 0,
                response_chars INTEGER DEFAULT 0,
                override TEXT,
                attempted TEXT,
                tool_calls INTEGER DEFAULT 0,
                reasoning_applied INTEGER DEFAULT 0,
                tool_dialect TEXT,
                call_role TEXT DEFAULT 'worker',
                router_decision TEXT,
                embed_dim INTEGER,
                agent TEXT,
                session TEXT,
                retries INTEGER DEFAULT 0
            )"""
        )
        c.execute("CREATE INDEX IF NOT EXISTS idx_ts ON calls(ts DESC)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_prov_ts ON calls(provider, ts DESC)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_role_ts ON calls(call_role, ts DESC)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_agent_ts ON calls(agent, ts DESC)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_session_ts ON calls(session, ts DESC)")
    # v4 columns are added by migration rather than being written into the
    # CREATE TABLE above, so there is exactly one code path that produces the
    # v4 shape and it is the same one an upgraded v3 database takes.
    migrate()


def log_call(
    provider,
    model,
    input_tokens=0,
    output_tokens=0,
    latency_ms=0,
    status="ok",
    error=None,
    prompt_chars=0,
    response_chars=0,
    override=None,
    attempted=None,
    cache_create_tokens=0,
    cache_read_tokens=0,
    tool_calls=0,
    reasoning_applied=False,
    tool_dialect=None,
    call_role="worker",
    router_decision=None,
    embed_dim=None,
    agent=None,
    session=None,
    retries=0,
    # ── v4 additions. All keyword-only-by-convention and all optional, so
    # every v3 call site keeps working untouched. ──
    tenant=None,
    project=None,
    user=None,
    usd=0.0,
    price_source=None,
    cache_hit=False,
    cache_kind=None,
    role=None,
    tier=None,
    escalations=0,
    tokens_saved=0,
    usd_saved=0.0,
) -> int:
    """Append one row to the ledger. Returns the new row id.

    v3 returned None; returning the id is additive (nothing inspected the
    return value) and lets the meter hand a row reference to the span.
    """
    with conn() as c:
        cur = c.execute(
            """INSERT INTO calls (ts, provider, model, input_tokens, output_tokens,
                                  cache_create_tokens, cache_read_tokens,
                                  latency_ms, status, error, prompt_chars, response_chars,
                                  override, attempted, tool_calls, reasoning_applied, tool_dialect,
                                  call_role, router_decision, embed_dim,
                                  agent, session, retries,
                                  tenant, project, "user", usd, price_source,
                                  cache_hit, cache_kind, role, tier, escalations,
                                  tokens_saved, usd_saved)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,
                       ?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                time.time(),
                provider,
                model,
                input_tokens,
                output_tokens,
                cache_create_tokens,
                cache_read_tokens,
                latency_ms,
                status,
                error,
                prompt_chars,
                response_chars,
                override,
                attempted,
                tool_calls,
                1 if reasoning_applied else 0,
                tool_dialect,
                call_role,
                router_decision,
                embed_dim,
                agent,
                session,
                retries,
                tenant,
                project,
                user,
                float(usd or 0.0),
                price_source,
                1 if cache_hit else 0,
                cache_kind,
                role,
                tier,
                int(escalations or 0),
                int(tokens_saved or 0),
                float(usd_saved or 0.0),
            ),
        )
        return int(cur.lastrowid or 0)


# ── v4 principal rollups ────────────────────────────────────────────────────


def _period_start(period: str, now: float | None = None) -> float:
    """Start of the current window. `lifetime` returns 0."""
    now = time.time() if now is None else now
    if period == "lifetime":
        return 0.0
    if period == "hour":
        return now - (now % 3600)
    if period == "minute":
        return now - (now % 60)
    if period == "month":
        import datetime as _dt

        d = _dt.datetime.fromtimestamp(now)
        return _dt.datetime(d.year, d.month, 1).timestamp()
    # default: calendar day, matching by_agent()'s bucketing
    return now - (now % 86400)


def spend_usd(dimension: str, value: str, since: float = 0.0, include_errors: bool = True) -> float:
    """Total dollars attributed to one principal since `since`.

    Spend is *derived from the ledger*, not tracked in a parallel counter.
    That means it survives a restart, cannot drift from what was actually
    logged, and there is no way to spend without the budget seeing it — the
    meter's ledger write is the same event the budget reads.
    """
    if dimension not in PRINCIPAL_DIMENSIONS:
        raise ValueError(
            f"unknown attribution dimension {dimension!r}; expected one of {PRINCIPAL_DIMENSIONS}"
        )
    q = f'SELECT COALESCE(SUM(usd), 0) AS total FROM calls WHERE "{dimension}" = ? AND ts >= ?'
    if not include_errors:
        q += " AND status='ok'"
    with conn() as c:
        row = c.execute(q, (value, since)).fetchone()
        return float(row["total"] or 0.0)


def by_principal(
    dimension: str = "agent",
    since: float | None = None,
    value: str | None = None,
    group_by_provider: bool = True,
) -> list[dict]:
    """Cost/token rollup grouped by one attribution dimension.

    A superset of `by_agent()`: same columns plus dollars and cache savings,
    and `dimension` may be any of PRINCIPAL_DIMENSIONS.
    """
    if dimension not in PRINCIPAL_DIMENSIONS:
        raise ValueError(
            f"unknown attribution dimension {dimension!r}; expected one of {PRINCIPAL_DIMENSIONS}"
        )
    args: list = [since if since is not None else _period_start("day")]
    where = ["ts >= ?", f'"{dimension}" IS NOT NULL']
    if value is not None:
        where.append(f'"{dimension}" = ?')
        args.append(value)
    group = f'"{dimension}"' + (", provider" if group_by_provider else "")
    select_provider = "provider, " if group_by_provider else "'(all)' AS provider, "
    q = (
        f'SELECT "{dimension}" AS principal_value, {select_provider}'
        "COUNT(*) AS calls, "
        "SUM(input_tokens) AS in_tok, SUM(output_tokens) AS out_tok, "
        "SUM(cache_read_tokens) AS cache_read_tok, "
        "SUM(cache_create_tokens) AS cache_create_tok, "
        "COALESCE(SUM(usd), 0) AS dollars, "
        "COALESCE(SUM(usd_saved), 0) AS dollars_saved, "
        "COALESCE(SUM(tokens_saved), 0) AS tokens_saved, "
        "SUM(COALESCE(cache_hit, 0)) AS cache_hits, "
        "SUM(latency_ms) AS total_latency_ms, "
        "SUM(retries) AS total_retries, "
        "SUM(CASE WHEN status='ok' THEN 1 ELSE 0 END) AS ok, "
        "SUM(CASE WHEN status='error' THEN 1 ELSE 0 END) AS errors "
        "FROM calls WHERE " + " AND ".join(where) + f" GROUP BY {group} ORDER BY dollars DESC"
    )
    with conn() as c:
        rows = [dict(r) for r in c.execute(q, args).fetchall()]
    for r in rows:
        r["dimension"] = dimension
        r["principal"] = f"{dimension}:{r['principal_value']}"
    return rows


def by_agent(session=None, since=None):
    where = ["ts >= ?"]
    # Day-rollover fix: bucket by calendar day, not by 24h window.
    args = [since if since is not None else (time.time() - (time.time() % 86400))]
    if session:
        where.append("session=?")
        args.append(session)
    q = (
        "SELECT agent, provider, COUNT(*) AS calls, "
        "SUM(input_tokens) AS in_tok, SUM(output_tokens) AS out_tok, "
        "SUM(latency_ms) AS total_latency_ms, "
        "SUM(retries) AS total_retries, "
        "SUM(CASE WHEN status='ok' THEN 1 ELSE 0 END) AS ok, "
        "SUM(CASE WHEN status='error' THEN 1 ELSE 0 END) AS errors "
        "FROM calls WHERE " + " AND ".join(where) + " AND agent IS NOT NULL "
        "GROUP BY agent, provider"
    )
    with conn() as c:
        rows = c.execute(q, args).fetchall()
        out: dict[str, list[dict]] = {}
        for r in rows:
            out.setdefault(r["agent"], []).append(dict(r))
        return out


def recent(limit=100, provider=None, status=None):
    q = "SELECT * FROM calls"
    where, args = [], []
    if provider:
        where.append("provider=?")
        args.append(provider)
    if status:
        where.append("status=?")
        args.append(status)
    if where:
        q += " WHERE " + " AND ".join(where)
    q += " ORDER BY ts DESC LIMIT ?"
    args.append(limit)
    with conn() as c:
        return [dict(r) for r in c.execute(q, args).fetchall()]


def aggregate(call_role=None):
    now = time.time()
    day_start = now - (now % 86400)
    q = """SELECT provider,
                  COUNT(*) AS calls,
                  SUM(CASE WHEN status='ok' THEN 1 ELSE 0 END) AS ok_calls,
                  SUM(CASE WHEN status='error' THEN 1 ELSE 0 END) AS errors,
                  SUM(input_tokens) AS in_tok,
                  SUM(output_tokens) AS out_tok,
                  SUM(cache_read_tokens) AS cache_reads,
                  SUM(cache_create_tokens) AS cache_creates,
                  SUM(tool_calls) AS tool_calls,
                  AVG(latency_ms) AS avg_latency,
                  MAX(ts) AS last_ts
             FROM calls WHERE ts >= ?"""
    args = [day_start]
    if call_role == "worker":
        q += " AND (call_role='worker' OR call_role IS NULL)"
    elif call_role == "router":
        q += " AND call_role LIKE 'router%'"
    elif call_role:
        q += " AND call_role=?"
        args.append(call_role)
    q += " GROUP BY provider"
    with conn() as c:
        rows = c.execute(q, args).fetchall()
        return {r["provider"]: dict(r) for r in rows}

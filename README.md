# glc_v4

`glc_v4` is the local Gateway for LLMs and Channels used by EAG V3. It owns provider credentials, model routing, rate limits, cost and audit records, voice, and channel adapters. Agent runtimes call it over HTTP; they do not import its provider code or read its keys.

v4 is `glc_v3` plus **agent economics**: the gateway now knows what a call costs, who it is billed to, and whether it is allowed to happen. Session 15 builds on it. Every v3 route, request field and response field still works unchanged — see `## What v4 adds` below.

## Requirements

- Python 3.11 or newer
- [`uv`](https://docs.astral.sh/uv/)
- At least one configured model provider
- Ollama only if you want local generation or gateway embeddings

## Install and run

```bash
uv sync
cp .env.example .env
# Edit .env locally. Never commit it.
uv run glc serve
```

The gateway listens on `http://127.0.0.1:8111` by default.

- Dashboard: `http://127.0.0.1:8111/`
- Help: `http://127.0.0.1:8111/help`
- OpenAPI: `http://127.0.0.1:8111/docs`
- Health: `http://127.0.0.1:8111/healthz`

## Multiple Gemini keys

Number the keys in `.env`:

```dotenv
GEMINI_API_KEY_1=replace-me
GEMINI_API_KEY_2=replace-me
GEMINI_API_KEY_3=replace-me
```

The gateway registers them as independently metered providers such as `gemini_1`, `gemini_2`, and `gemini_3`. A caller can request the logical provider `gemini`; the router selects an available numbered slot. The dashboard shows each slot separately.

## Smoke test

```bash
curl -s http://127.0.0.1:8111/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "Say hello in one line."}],
    "provider": "gemini",
    "max_tokens": 80,
    "temperature": 0
  }'
```

The response reports the actual provider slot and model used.

## What v4 adds

Five modules and four config files. Nothing here is required: with the shipped
configuration the gateway behaves exactly like `glc_v3`, and every feature is
armed by editing YAML rather than Python.

| Module | Does |
|---|---|
| `glc/economics/pricing.py` | Per-**model** pricing from `pricing.yaml`, with cache-read/cache-write and batch multipliers as data. v3's per-provider table survives as the fallback, so an unpriced model reports what v3 reported. |
| `glc/economics/meter.py` | Attribution across five dimensions — **tenant, project, user**, agent, session. One priced ledger row per call. |
| `glc/economics/budget.py` | The **hard controller**. Admission on a projected worst-case cost *before* the provider is called; breach → HTTP 402 with the numbers. |
| `glc/telemetry/otel.py` | OTel `chat` spans with `gen_ai.*` attributes plus computed cost. Content capture **off** by default. |
| `glc/routing/policy.py` | Role→tier policy, cost/quality candidate ordering, a servable `HUGE` tier, cascade escalation. |
| `glc/cache/semantic.py` | Response cache: embed the request, cosine-match, and on a hit **skip the provider call entirely**. |

### Config, not code

| File | Packaged default | Override with |
|---|---|---|
| `pricing.yaml` | `glc/economics/pricing.yaml` | `~/.glc/pricing.yaml` or `GLC_PRICING_YAML` |
| `budgets.yaml` | `glc/economics/budgets.yaml` (ships **empty** — nothing is refused) | `~/.glc/budgets.yaml` or `GLC_BUDGETS_YAML` |
| `routing.yaml` | `glc/routing/routing.yaml` | `~/.glc/routing.yaml` or `GLC_ROUTING_YAML` |
| `cache.yaml` | `glc/cache/cache.yaml` (semantic cache ships **opt-in**) | `~/.glc/cache.yaml` or `GLC_CACHE_YAML` |

Adding a model, a role, a tier or a budget is an edit to one of those files.
There is no Python change and no list of names inside the library. A malformed
budget or routing file raises rather than silently degrading into "no budget" —
`GET /v1/budget` and `app.state.config_errors` report what failed.

### New endpoints

```
GET    /v1/budget                 every loaded policy
GET    /v1/budget/{principal}     limit, spend, remaining — e.g. /v1/budget/session:run-42
POST   /v1/budget                 arm or move a ceiling at runtime (limit_usd: 0 = stop now)
DELETE /v1/budget/{principal}     drop a runtime override
GET    /v1/cost/by_principal      five-dimension rollup (superset of /v1/cost/by_agent)
GET    /v1/cache/stats            hit rate, tokens and dollars saved
GET    /v1/pricing                the resolved price table, or one (provider, model)
GET    /v1/telemetry              tracer state and exporters
POST   /v1/cache/purge            drop expired (or all) cache entries
```

`POST /v1/chat` keeps its contract. New **optional** request fields:
`tenant`, `project`, `user`, `semantic_cache`, `batch`, `cost_quality_tradeoff`,
`escalate`. New response fields `cost`, `budget`, `cache`, `trace` are `null`
unless the corresponding feature ran.

### Arming the budget controller

```bash
# ceiling for one run, enforced before any provider is contacted
curl -s -X POST http://127.0.0.1:8111/v1/budget \
  -H 'Content-Type: application/json' \
  -d '{"principal": "session:run-42", "limit_usd": 0.50, "period": "lifetime"}'

# a call whose projected cost does not fit gets HTTP 402 and is never sent
curl -s -X POST http://127.0.0.1:8111/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "...", "provider": "gemini", "max_tokens": 2048, "session": "run-42"}'
```

Budgets are enforced in code, not stated in a prompt. Token elasticity means a
model asked to stay under a limit does not, so the ceiling lives where the model
cannot argue with it.

### Tracing

```bash
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 uv run glc     # export to Jaeger
GLC_OTEL_CONSOLE=1 uv run glc                                    # dump spans to stdout
GLC_OTEL_CAPTURE_CONTENT=1 ...                                   # attach prompts (PII — off by default)
```

With no endpoint set, spans are a no-op and no collector is needed.

### Ledger migration

The `calls` table gains twelve nullable columns (`tenant`, `project`, `user`,
`usd`, `cache_hit`, …) through `ALTER TABLE ADD COLUMN`. A v3 database is
upgraded in place on boot: existing rows keep their meaning and every v3 query
still answers. Spend is read back out of this ledger rather than tracked in a
parallel counter, so it survives a restart and cannot drift from what was billed.

## Relationship to Session 13

`glc_v4` is a dependency of `S13Code`/`S15Code`, not its parent project. The ownership boundary is deliberate:

| `glc_v3` owns | `S13Code` owns |
|---|---|
| Keys, providers and models | Live task graph |
| Routing, quotas and costs | Memory and semantic indexing |
| Channels and voice | A2A discovery and delegation |
| `/v1/chat` | `/v1/agent/*` |

`glc_v3` must return `404` for Session 13 agent routes. `S13Code` must return `404` for gateway model routes.

## Development

```bash
uv run ruff check .
uv run pytest -q
```

Never commit `.env`, API keys, local databases, audit records, pairing state, or user memory.

## License

MIT. See `LICENSE`.

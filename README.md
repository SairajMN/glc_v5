# glc_v3

`glc_v3` is the local Gateway for LLMs and Channels used by EAG V3. It owns provider credentials, model routing, rate limits, cost and audit records, voice, and channel adapters. Agent runtimes call it over HTTP; they do not import its provider code or read its keys.

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

## Relationship to Session 13

`glc_v3` is a dependency of `S13Code`, not its parent project. The ownership boundary is deliberate:

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

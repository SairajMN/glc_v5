# Voice guide

Session 11 §9 names the current free-tier picks for STT and TTS in
mid-2026. This guide covers the API keys, model downloads, and
gotchas per provider.

## STT (POST /v1/transcribe)

| `prefer`     | Provider                       | Setup                                                     |
|--------------|--------------------------------|-----------------------------------------------------------|
| `default`    | Groq Whisper Large v3 Turbo    | Free signup at `console.groq.com`; export `GROQ_API_KEY`. |
| `local`      | whisper.cpp + base model       | `./daemon/install.sh --models` fetches the base model.    |
| `streaming`  | (intentionally returns 400)    | Streaming STT belongs on the Gemini Live WebSocket; S12.  |

The Groq free tier covers thousands of minutes per month; the model
runs on Groq's LPU hardware so latency stays sub-second for short
clips. Local whisper.cpp is the offline fallback at ~200 MB on disk.

## TTS (POST /v1/speak)

| `prefer`     | Provider             | Setup                                                                                    |
|--------------|----------------------|------------------------------------------------------------------------------------------|
| `default`    | Kokoro-82M (local)   | `uv pip install kokoro`; model lazy-loads to `~/.glc/models/kokoro-82M/`.                |
| `quality`    | ElevenLabs Flash v2.5| Free tier 10k chars/month; export `ELEVENLABS_API_KEY` and `ELEVENLABS_VOICE_ID`.        |
| `streaming`  | Cartesia Sonic       | Free tier; export `CARTESIA_API_KEY` and (optional) `CARTESIA_VOICE_ID`.                 |
| `fallback`   | system TTS           | macOS `say`, elsewhere `pyttsx3`. Zero-config; useful when offline and no Kokoro model.  |

Kokoro is the daily driver: open weights, 82M parameters, runs
faster than realtime on a laptop CPU, voice quality close to
ElevenLabs for most uses. ElevenLabs covers the cases where Kokoro's
voice palette is too narrow. Cartesia is the latency play —
sub-50ms time-to-first-audio for streaming use cases like
`twilio_voice` outbound.

## Realtime full-duplex

`webui` and `twilio_voice` adapters open a Gemini Live WebSocket
session through the gateway. This route is **not** implemented in
S11 — the adapter test suite mocks it. The real session token is
deliverable in S12.

## Cost reporting

STT and TTS calls land in the per-agent ledger under
`agent: channel:<name>`. View with:

```sh
curl http://localhost:8111/v1/cost/by_agent
```

## Offline tests

The voice routing tests under `tests/test_voice_routing.py` inject
fakes via `monkeypatch` and never hit a real provider. CI runs
offline and passes on a fresh checkout with no API keys set.

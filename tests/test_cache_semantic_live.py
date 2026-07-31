"""Semantic cache against the REAL embedder, not a stub.

``test_cache_semantic.py`` is the arithmetic suite: it injects a fixed embedder
and checks cosine, the threshold, the namespace hash and the opt-in rule. Every
one of those tests passes whether or not nomic is actually wired up, because
none of them ever computes an embedding. That is the right design for testing
the maths and the wrong thing to rely on for believing the cache works.

These tests close that gap. They run the cache through
:func:`glc.cache.semantic.build_semantic_cache`, which is the same wiring
``glc/main.py`` uses at startup — the gateway's embedder failover ring, the
768-dim pinning, nomic's task prefix — and assert on vectors that came out of
Ollama.

They skip, loudly and cleanly, when Ollama is not answering, so CI without a
local model server stays green. A skip here is not a pass: it means the real
path was not exercised, and the ``@pytest.mark.requires_models`` marker is there
so a run can demand them.
"""

from __future__ import annotations

import os

import httpx
import pytest

from glc import db
from glc import embedders as E
from glc.cache.semantic import SemanticCacheConfig, build_semantic_cache, cosine
from glc.economics.meter import Principal

pytestmark = pytest.mark.requires_models

#: Read the same way ``glc/embedders.py`` reads them, so a deployment that
#: points its ring somewhere else tests the ring it actually runs.
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL = os.getenv("EMBED_OLLAMA_MODEL", "nomic-embed-text")


def _ollama_has_model() -> tuple[bool, str]:
    """Is a local Ollama serving the embedding model the ring is configured for?"""
    try:
        response = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        response.raise_for_status()
    except Exception as error:  # pragma: no cover - environment dependent
        return False, f"ollama at {OLLAMA_URL} is not answering ({type(error).__name__})"
    names = [m.get("name", "") for m in response.json().get("models", [])]
    if not any(name.split(":")[0] == MODEL for name in names):
        return False, f"ollama has no {MODEL} (has {names[:6]})"
    return True, ""


_AVAILABLE, _WHY = _ollama_has_model()
requires_ollama = pytest.mark.skipif(not _AVAILABLE, reason=_WHY or "ollama unavailable")


@pytest.fixture(autouse=True)
def _fresh():
    db.init()
    yield


@pytest.fixture
def cache():
    """The cache wired exactly as the gateway wires it, on the real ring.

    ``ttl_seconds`` and ``max_entries`` are relaxed and the scope is left at the
    default so these tests exercise the shipped namespace behaviour rather than
    a bespoke configuration.
    """
    ring = [E.OllamaEmbedder(MODEL, OLLAMA_URL)]
    built = build_semantic_cache(ring)
    built.config = SemanticCacheConfig(
        enabled=True,
        default_on=True,
        threshold=built.config.threshold,
        ttl_seconds=0,
        max_entries=0,
        namespace_fields=list(built.config.namespace_fields),
        scope_dimensions=list(built.config.scope_dimensions),
        task_type=built.config.task_type,
    )
    return built


# ── the embedding is real ───────────────────────────────────────────────────


@requires_ollama
async def test_a_real_embedding_is_768_dimensional(cache):
    """The dimension both providers are pinned to, from the actual model.

    The stub suite's vectors are 2- and 3-dimensional. This is the assertion
    that fails the moment the ring stops reaching nomic.
    """
    vector = await cache.embed("What is the largest planet in the solar system?")
    assert vector is not None, "the real embedder returned nothing"
    assert len(vector) == E.EMBED_DIM == 768
    assert all(isinstance(x, float) for x in vector)


@requires_ollama
async def test_a_real_embedding_is_not_a_constant_vector(cache):
    """A stub that returns one fixed vector matches everything at cosine 1.0.

    Three properties separate a real embedder from that: it is deterministic on
    the same text, it is NOT identical on different text, and its components are
    genuinely distinct rather than a handful of repeated values.
    """
    once = await cache.embed("How do I reset my password?")
    twice = await cache.embed("How do I reset my password?")
    other = await cache.embed("Give me a recipe for pancakes.")

    assert cosine(once, twice) > 0.9999
    assert cosine(once, other) < 0.99
    assert len({round(x, 9) for x in once}) > 100


@requires_ollama
async def test_the_task_prefix_nomic_requires_is_actually_applied(cache):
    """nomic-embed-text wants ``search_query:`` / ``search_document:``.

    ``OllamaEmbedder`` adds it, so embedding the same text under the two task
    types must give two different vectors. If the prefix were dropped they would
    be identical, and retrieval quality would quietly degrade with no error.
    """
    text = "What is the boiling point of water at sea level?"
    as_query = (await E.embed_with_failover([E.OllamaEmbedder(MODEL, OLLAMA_URL)],
                                            text, "retrieval_query"))[1]["embedding"]
    as_document = (await E.embed_with_failover([E.OllamaEmbedder(MODEL, OLLAMA_URL)],
                                               text, "retrieval_document"))[1]["embedding"]
    assert len(as_query) == len(as_document) == 768
    assert cosine(as_query, as_document) < 0.9999


# ── the cache decides correctly on real vectors ─────────────────────────────


@requires_ollama
async def test_a_real_paraphrase_hits(cache):
    """The whole point of the cache, measured rather than stubbed.

    The pair is the one p6 measures at cosine 0.957325 through the live gateway.
    The assertion is against the cache's own configured threshold, so lowering
    ``threshold`` in cache.yaml cannot make this test start lying.
    """
    fields = {"model": "test-model", "system": "answer briefly"}
    await cache.store(
        "Which planet in the solar system is the largest?",
        fields,
        {"text": "Jupiter."},
        provider="p",
        model="test-model",
        input_tokens=20,
        output_tokens=7,
        usd=1.925e-05,
    )
    found = await cache.lookup("What is the biggest planet in our solar system?", fields)
    assert found.hit is True, f"paraphrase missed at similarity {found.best_similarity:.6f}"
    assert found.similarity >= cache.config.threshold
    assert found.response == {"text": "Jupiter."}
    assert found.tokens_saved == 27
    assert found.usd_saved == pytest.approx(1.925e-05)


@requires_ollama
async def test_a_real_unrelated_prompt_misses(cache):
    """Two questions with nothing in common must not collide."""
    fields = {"model": "test-model"}
    await cache.store(
        "What does HTTP status code 404 mean?", fields, {"text": "Not Found."},
        provider="p", model="test-model",
    )
    found = await cache.lookup("Give me a recipe for pancakes.", fields)
    assert found.hit is False
    # `best_similarity` is the honest counterweight the stats endpoint reports:
    # a miss still says how close it got, and this one must not be close.
    assert found.best_similarity < cache.config.threshold


@requires_ollama
async def test_the_similarity_the_cache_acts_on_is_the_cosine_of_the_two_vectors(cache):
    """No hidden rescaling between the embedder and the verdict.

    Embed both prompts here, compute the cosine here, and check the number the
    cache reported matches. Without this, a reported similarity is hearsay.
    """
    a, b = "How do I reset my password?", "What are the steps to reset my password?"
    fields = {"model": "test-model"}
    await cache.store(a, fields, {"text": "Use the reset link."}, provider="p", model="test-model")
    found = await cache.lookup(b, fields)
    expected = cosine(await cache.embed(a), await cache.embed(b))
    reported = found.similarity if found.hit else found.best_similarity
    assert reported == pytest.approx(expected, abs=1e-9)


@requires_ollama
async def test_argument_order_does_not_survive_the_embedding(cache):
    """The measured failure mode, pinned here so nobody forgets it.

    "London to New York" and "New York to London" are the same bag of words and
    opposite requests. nomic scores them at ~0.997 — higher than any paraphrase
    S15's p6 measured — so at ANY threshold in the production range the cache
    serves one from the other. That is not a bug in this code and there is no
    threshold that fixes it: a similarity model that ignores argument order
    cannot be asked to respect it.

    The assertion is deliberately the wrong-looking one. It documents what the
    cache DOES, so that a future embedder change shows up as a failing test and
    a decision, rather than as a silently different production behaviour.
    """
    fields = {"model": "test-model"}
    await cache.store(
        "Show me flights from London to New York.", fields, {"text": "LHR -> JFK, 3 options."},
        provider="p", model="test-model",
    )
    found = await cache.lookup("Show me flights from New York to London.", fields)
    assert found.hit is True, (
        f"the reversed-direction prompt no longer collides (similarity "
        f"{found.best_similarity:.6f} < threshold {cache.config.threshold}). The embedder or the "
        f"threshold changed; re-run S15's p6 and update the lecture's numbers."
    )
    assert found.similarity > 0.99
    assert found.response == {"text": "LHR -> JFK, 3 options."}


@requires_ollama
async def test_an_antonym_swap_is_correctly_rejected(cache):
    """Not every near-miss collides: a one-word antonym is caught.

    Largest/smallest scores ~0.78, far under any production threshold. Pairing
    this with the test above is the honest picture — the cache fails on word
    ORDER, not on word CHOICE.
    """
    fields = {"model": "test-model"}
    await cache.store(
        "What is the largest planet in the solar system?", fields, {"text": "Jupiter."},
        provider="p", model="test-model",
    )
    found = await cache.lookup("What is the smallest planet in the solar system?", fields)
    assert found.hit is False
    assert found.best_similarity < 0.9


# ── isolation still holds when the vectors are real ─────────────────────────


@requires_ollama
async def test_a_different_model_misses_even_on_a_real_near_identical_prompt(cache):
    """Namespace beats similarity. A cached answer from another model is not
    the same answer, however close the two questions are."""
    stored = {"model": "cheap-model", "system": "answer briefly"}
    await cache.store(
        "Which planet in the solar system is the largest?", stored, {"text": "Jupiter."},
        provider="p", model="cheap-model",
    )
    query = "What is the biggest planet in our solar system?"
    assert (await cache.lookup(query, stored)).hit is True
    assert (await cache.lookup(query, {**stored, "model": "expensive-model"})).hit is False
    assert (await cache.lookup(query, {**stored, "system": "answer at length"})).hit is False


@requires_ollama
async def test_a_different_tenant_misses_on_a_real_paraphrase(cache):
    """`scope_dimensions` is the cross-tenant seal, checked on real vectors."""
    cache.config.scope_dimensions = ["tenant"]
    fields = {"model": "test-model"}
    await cache.store(
        "Which planet in the solar system is the largest?", fields, {"text": "Jupiter."},
        provider="p", model="test-model", principal=Principal(tenant="acme"),
    )
    query = "What is the biggest planet in our solar system?"
    assert (await cache.lookup(query, fields, principal=Principal(tenant="acme"))).hit is True
    assert (await cache.lookup(query, fields, principal=Principal(tenant="other"))).hit is False


@requires_ollama
async def test_a_real_run_records_no_embed_failures(cache):
    """`embed_failures` counts every embedding the cache could not get.

    It is the counter that stays at zero when the ring is healthy and climbs
    silently when it is not, because the cache treats an embedding failure as a
    skip rather than an error. A real round trip must leave it alone.
    """
    before = cache.stats()["embed_failures"]
    fields = {"model": "test-model"}
    await cache.store("What is the capital of Australia?", fields, {"text": "Canberra."},
                      provider="p", model="test-model")
    await cache.lookup("What is the capital of Austria?", fields)
    assert cache.stats()["embed_failures"] == before

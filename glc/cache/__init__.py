"""Cache package.

`glc/cache.py` (Gemini explicit prompt caching) became `glc/cache/gemini.py`
in v4 so the semantic/response cache could sit beside it. `GeminiCache` is
re-exported here so `from glc.cache import GeminiCache` keeps working.

Two different caches, deliberately not merged:

* `gemini.GeminiCache` — byte-exact prompt caching. The provider still runs the
  call; the discount is on the cached *prefix tokens*.
* `semantic.SemanticCache` — embedding-similarity response cache. On a hit the
  provider call does not happen at all, so the saving is 100% of the tokens.
"""

from __future__ import annotations

from glc.cache.gemini import GeminiCache
from glc.cache.semantic import (
    SemanticCache,
    SemanticCacheConfig,
    cosine,
    load_cache_config,
)

__all__ = [
    "GeminiCache",
    "SemanticCache",
    "SemanticCacheConfig",
    "cosine",
    "load_cache_config",
]

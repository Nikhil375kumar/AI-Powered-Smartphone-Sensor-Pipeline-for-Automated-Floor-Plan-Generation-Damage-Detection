"""
src/utils/groq_client.py
~~~~~~~~~~~~~~~~~~~~~~~~
Thin wrapper around the Groq SDK (free tier).

Used for:
  - LLM-assisted damage classification when YOLO confidence is low
  - Photo-tier scene description to improve depth scale estimation
  - Concealed damage reasoning fallback

Gracefully no-ops if GROQ_API_KEY is not set — the pipeline runs
fully without it; Groq is a quality-boost, not a hard dependency.

Get a FREE API key at: https://console.groq.com  (no credit card)
"""

from __future__ import annotations

import os
from typing import Optional

from src.utils.logger import get_logger

log = get_logger(__name__)


class GroqClient:
    """
    Lightweight Groq API client.

    Args:
        api_key:    Groq API key. Reads ``GROQ_API_KEY`` from env if not provided.
        model:      Model ID. Reads ``GROQ_MODEL`` from env, defaults to llama3-70b-8192.
    """

    # Groq models available on developer (free) plan — tried in order.
    # Ref: https://console.groq.com/docs/models
    _FALLBACK_MODELS = [
        "openai/gpt-oss-20b",     # 1000 T/s, cheapest, fastest
        "openai/gpt-oss-120b",    # 500 T/s, higher quality
        "llama-3.1-8b-instant",   # enterprise, may work on some plans
        "llama-3.3-70b-versatile",# enterprise, may work on some plans
    ]
    _MAX_CONSECUTIVE_FAILURES = 3  # disable Groq after this many back-to-back failures

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        self.api_key: str = api_key or os.getenv("GROQ_API_KEY", "").strip()
        self.model: str = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
        self._client = None
        self._cache: dict = {}        # class_name → refined_class_name
        self._fail_count: int = 0     # consecutive failure counter
        self._disabled: bool = False  # set True after too many failures

        if self.api_key:
            self._init_client()
        else:
            log.warning(
                "GROQ_API_KEY not set — Groq features disabled. "
                "Get a free key at https://console.groq.com"
            )

    def _init_client(self) -> None:
        """Initialise the Groq SDK client."""
        try:
            from groq import Groq  # type: ignore
            self._client = Groq(api_key=self.api_key)
            log.info("Groq client ready (model={})", self.model)
        except ImportError:
            log.warning(
                "groq SDK not installed — run: pip install groq  "
                "(Groq features will be disabled)"
            )
            self._client = None

    # ── Public ────────────────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        """True if the Groq client is ready AND has not been auto-disabled."""
        return self._client is not None and not self._disabled

    def chat(
        self,
        prompt: str,
        system: str = "You are an expert construction damage assessor.",
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> Optional[str]:
        """
        Send a chat completion request to Groq.

        Args:
            prompt:      User message to send.
            system:      System instruction for the model.
            max_tokens:  Maximum tokens in the response.
            temperature: Sampling temperature (lower = more deterministic).

        Returns:
            Response text string, or None if unavailable / error.
        """
        if not self.available:
            return None

        # Try the configured model first, then fallbacks
        models_to_try = [self.model] + [
            m for m in self._FALLBACK_MODELS if m != self.model
        ]
        for attempt_model in models_to_try:
            try:
                response = self._client.chat.completions.create(
                    model=attempt_model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                if attempt_model != self.model:
                    log.info("Switched Groq model: {} → {}", self.model, attempt_model)
                    self.model = attempt_model
                self._fail_count = 0  # reset on success
                return response.choices[0].message.content
            except Exception as exc:
                err_str = str(exc)
                if any(k in err_str for k in ("decommissioned", "model_not_found", "not supported", "not found")):
                    log.debug("Model '{}' unavailable, trying next …", attempt_model)
                    continue
                # Non-model error (rate limit, network, etc.) — count as failure
                self._fail_count += 1
                if self._fail_count >= self._MAX_CONSECUTIVE_FAILURES:
                    self._disabled = True
                    log.warning(
                        "Groq disabled after {} consecutive failures — "
                        "pipeline continues without LLM refinement.",
                        self._fail_count,
                    )
                else:
                    log.debug("Groq API error (failure {}/{}): {}",
                              self._fail_count, self._MAX_CONSECUTIVE_FAILURES, exc)
                return None
        # All models exhausted (all decommissioned/not found)
        self._disabled = True
        log.warning("All Groq models unavailable on this plan — Groq disabled.")
        return None

    def classify_damage(self, detected_class: str, context: str = "") -> str:
        """
        Ask Groq to confirm or refine a YOLO damage classification.
        Caches results — same class name is never sent twice.
        """
        if not self.available:
            return detected_class

        # Cache hit — avoid duplicate API calls for same class
        cache_key = detected_class.lower()
        if cache_key in self._cache:
            return self._cache[cache_key]

        prompt = (
            f"A computer vision model detected '{detected_class}' on an interior wall. "
            f"Context: {context or 'no additional context'}. "
            "Confirm the most appropriate damage category from: "
            "[crack, water_stain, mold, peeling_paint, structural_damage, efflorescence, none]. "
            "Reply with ONLY the category name, nothing else."
        )
        result = self.chat(prompt, max_tokens=20, temperature=0.1)
        refined = result.strip().lower() if result else detected_class
        self._cache[cache_key] = refined  # cache for next call
        return refined


# ── Module-level singleton (lazy) ─────────────────────────────────────────────
_groq: Optional[GroqClient] = None


def get_groq_client() -> GroqClient:
    """Return the module-level singleton GroqClient (created on first call)."""
    global _groq
    if _groq is None:
        _groq = GroqClient()
    return _groq

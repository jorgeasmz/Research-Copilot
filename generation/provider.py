"""The generation backend, behind an interface the rest of the package depends on."""

import logging
from collections.abc import Iterator
from typing import Protocol

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from generation import config

logger = logging.getLogger(__name__)


EXHAUSTED = 429


def _code(error: BaseException) -> int | None:
    return getattr(error, "code", None)


def _retryable(error: BaseException) -> bool:
    """A 5xx is the model being briefly busy, which waiting can fix."""
    code = _code(error)
    return code is not None and 500 <= code < 600


def _switchable(error: BaseException) -> bool:
    """
    Whether another model is worth trying.

    The free tier meters requests per day per model, so an exhausted quota is
    permanent for that model today and immediately fixable by using another.
    Waiting on it, as the retry does for a 5xx, would waste the whole window.
    """
    return _retryable(error) or _code(error) == EXHAUSTED


transient_retry = retry(
    retry=retry_if_exception(_retryable),
    stop=stop_after_attempt(config.RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    reraise=True,
)


class Provider(Protocol):
    """What the pipeline needs from a language model."""

    def stream(self, prompt: str) -> Iterator[str]:
        """Yields the answer in fragments, as they are produced."""
        ...

    def complete(self, prompt: str) -> str:
        """Returns the whole answer. Used where streaming buys nothing."""
        ...


class GeminiProvider:
    """Google AI Studio, whose free tier meters requests rather than tokens."""

    def __init__(
        self,
        model: str = config.MODEL,
        api_key: str = "",
        fallbacks: list[str] | None = None,
    ):
        from google import genai

        key = api_key or config.API_KEY
        if not key:
            raise RuntimeError("GOOGLE_API_KEY is not set")

        self.models = [model, *(config.FALLBACK_MODELS if fallbacks is None else fallbacks)]
        self.client = genai.Client(api_key=key)

    def _over_chain(self, call):
        """
        Tries each model in turn, moving on when one is saturated.

        Retrying a busy model is worth a few seconds; after that the only thing
        that helps is asking a different one.
        """
        last = None
        for name in self.models:
            try:
                return call(name)
            except Exception as error:
                if not _switchable(error):
                    raise
                logger.warning("%s unavailable (%s), trying the next model", name, _code(error))
                last = error
        raise last

    def _settings(self) -> dict:
        return {
            "temperature": config.TEMPERATURE,
            "max_output_tokens": config.MAX_OUTPUT_TOKENS,
        }

    @transient_retry
    def _open_stream(self, name: str, prompt: str):
        """
        Opens the stream and pulls the first chunk inside the guarded call.

        The client issues the request lazily, so without forcing one chunk a
        saturated model would fail on first iteration, outside the fallback.
        """
        chunks = self.client.models.generate_content_stream(
            model=name, contents=prompt, config=self._settings()
        )
        return next(chunks, None), chunks

    @transient_retry
    def _complete_once(self, name: str, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=name, contents=prompt, config=self._settings()
        )
        return response.text or ""

    def stream(self, prompt: str) -> Iterator[str]:
        first, rest = self._over_chain(lambda name: self._open_stream(name, prompt))
        for chunk in (first, *rest) if first is not None else rest:
            if chunk.text:
                yield chunk.text

    def complete(self, prompt: str) -> str:
        return self._over_chain(lambda name: self._complete_once(name, prompt))


def build(name: str = config.PROVIDER) -> Provider:
    if name == "gemini":
        return GeminiProvider()
    raise ValueError(f"unknown provider: {name}")

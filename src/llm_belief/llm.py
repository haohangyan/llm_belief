"""Small OpenAI-compatible interface for local or H200-hosted models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from openai import OpenAI


class LLMClient:
    """Chat-completion client shared by local development and vLLM on H200."""

    def __init__(self, model: str) -> None:
        self.model = model
        self._client = OpenAI(
            base_url="http://127.0.0.1:8000/v1",
            api_key="not-needed",
        )

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[dict(message) for message in messages],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("The model returned no text")
        return content

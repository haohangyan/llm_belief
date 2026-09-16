"""Small OpenAI-compatible interface for local or H200-hosted models."""

import json
import logging
import os

from openai import OpenAI


logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpx2").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


class LLMClient:
    """Chat-completion client shared by local development and vLLM on H200."""

    def __init__(
        self,
        model,
        max_tokens=1024,
        reasoning_effort=None,
        base_url="http://127.0.0.1:8000/v1",
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self._client = OpenAI(
            base_url=base_url,
            api_key="not-needed",
        )

    def complete(self, messages, temperature=0.0, max_tokens=None):
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

    def complete_json(self, messages, name, schema):
        for attempt in range(2):
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[dict(message) for message in messages],
                temperature=0.0,
                max_tokens=self.max_tokens,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": name,
                        "schema": schema,
                        "strict": True,
                    },
                },
                extra_body=(
                    {"reasoning_effort": self.reasoning_effort}
                    if self.reasoning_effort
                    else None
                ),
            )
            content = response.choices[0].message.content
            if content:
                return json.loads(content), {}
            if attempt == 0:
                print("The model returned no JSON; retrying once", flush=True)
        raise RuntimeError("The model returned no JSON after one retry")


class OpenAILLMClient:
    """OpenAI Responses API client using OPENAI_API_KEY."""

    def __init__(self, model="gpt-5.6-luna"):
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.model = model
        self._client = OpenAI()

    def complete_json(self, messages, name, schema):
        response = self._client.responses.create(
            model=self.model,
            input=[dict(message) for message in messages],
            reasoning={"effort": "low"},
            text={
                "format": {
                    "type": "json_schema",
                    "name": name,
                    "schema": schema,
                    "strict": True,
                }
            },
            store=False,
        )
        return json.loads(response.output_text), {}

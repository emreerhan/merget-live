from __future__ import annotations

import asyncio
import json
import random
import re
import time
from pathlib import Path
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel

from .config import OpenRouterConfig
from .provenance import canonical_hash, write_json_atomic
from .security import read_api_key

ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


class OpenRouterError(RuntimeError):
    pass


class NonRetryableOpenRouterError(OpenRouterError):
    pass


def strict_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Adapt Pydantic output to provider strict-structured-output rules."""
    result = json.loads(json.dumps(schema))
    definitions = result.get("$defs", {})

    def visit(node: Any) -> Any:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                name = ref.rsplit("/", 1)[-1]
                if name not in definitions:
                    raise ValueError(f"unresolved JSON-schema reference: {ref}")
                resolved = json.loads(json.dumps(definitions[name]))
                return visit(resolved)
            properties = node.get("properties")
            if isinstance(properties, dict):
                node["required"] = list(properties)
                node["additionalProperties"] = False
            elif node.get("type") == "object":
                # OpenAI strict structured outputs do not accept arbitrary-key
                # mappings. Generated provenance and optional quote maps are
                # therefore constrained to an empty object and populated
                # deterministically after validation.
                node["properties"] = {}
                node["required"] = []
                node["additionalProperties"] = False
            node.pop("default", None)
            node.pop("$defs", None)
            for key, value in list(node.items()):
                node[key] = visit(value)
        elif isinstance(node, list):
            return [visit(value) for value in node]
        return node

    return visit(result)


class OpenRouterClient:
    def __init__(
        self,
        config: OpenRouterConfig,
        key_file: Path,
        cache_dir: Path,
        *,
        transport: httpx.BaseTransport | None = None,
    ):
        self.config = config
        self.key_file = key_file
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._transport = transport
        self.usage = {"requests": 0, "cache_hits": 0, "input_tokens": 0, "output_tokens": 0, "cost": 0.0}

    def _cache_key(
        self,
        *,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        purpose: str,
    ) -> str:
        return canonical_hash(
            {
                "model": self.config.model,
                "reasoning": self.config.reasoning_effort,
                "messages": messages,
                "schema": schema,
                "purpose": purpose,
            }
        )

    def complete_structured(
        self,
        *,
        messages: list[dict[str, str]],
        response_model: type[ResponseModel],
        purpose: str,
        force: bool = False,
    ) -> ResponseModel:
        schema = strict_json_schema(response_model.model_json_schema())
        cache_key = self._cache_key(messages=messages, schema=schema, purpose=purpose)
        cache_path = self.cache_dir / purpose / f"{cache_key}.json"
        if cache_path.is_file() and not force:
            self.usage["cache_hits"] += 1
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            return response_model.model_validate(cached["parsed"])

        api_key = read_api_key(self.key_file)
        payload = {
            "model": self.config.model,
            "messages": messages,
            "reasoning": {
                "effort": self.config.reasoning_effort,
                "exclude": self.config.exclude_reasoning,
            },
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": re.sub(r"[^a-zA-Z0-9_-]", "_", purpose)[:64],
                    "strict": True,
                    "schema": schema,
                },
            },
            "provider": {"require_parameters": True},
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost/entireio-retrieval",
            "X-Title": "entireio-cli retrieval research",
        }
        error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                with httpx.Client(
                    timeout=self.config.timeout_seconds, transport=self._transport
                ) as client:
                    response = client.post(
                        f"{self.config.base_url.rstrip('/')}/chat/completions",
                        headers=headers,
                        json=payload,
                    )
                if response.status_code in {408, 409, 429} or response.status_code >= 500:
                    raise OpenRouterError(
                        f"retryable OpenRouter response {response.status_code}: {response.text[:500]}"
                    )
                if response.status_code >= 400:
                    raise NonRetryableOpenRouterError(
                        f"OpenRouter response {response.status_code}: {response.text[:2000]}"
                    )
                response.raise_for_status()
                body = response.json()
                content = body["choices"][0]["message"]["content"]
                parsed_json = json.loads(content)
                parsed = response_model.model_validate(parsed_json)
                usage = body.get("usage", {})
                self.usage["requests"] += 1
                self.usage["input_tokens"] += int(usage.get("prompt_tokens") or 0)
                self.usage["output_tokens"] += int(usage.get("completion_tokens") or 0)
                self.usage["cost"] += float(usage.get("cost") or 0.0)
                write_json_atomic(
                    cache_path,
                    {
                        "cache_key": cache_key,
                        "model": body.get("model", self.config.model),
                        "purpose": purpose,
                        "parsed": parsed.model_dump(mode="json"),
                        "usage": usage,
                    },
                )
                return parsed
            except NonRetryableOpenRouterError as exc:
                error = exc
                break
            except (httpx.HTTPError, KeyError, ValueError, OpenRouterError) as exc:
                error = exc
                if attempt >= self.config.max_retries:
                    break
                delay = min(30.0, 2**attempt + random.random())
                time.sleep(delay)
        raise OpenRouterError(
            f"OpenRouter {purpose} failed after {self.config.max_retries + 1} attempts: {error}"
        )

    async def complete_many(
        self,
        requests: list[dict[str, Any]],
    ) -> list[BaseModel | Exception]:
        semaphore = asyncio.Semaphore(self.config.max_concurrency)

        async def one(request: dict[str, Any]):
            async with semaphore:
                try:
                    return await asyncio.to_thread(self.complete_structured, **request)
                except Exception as exc:  # caller persists the non-secret failure
                    return exc

        return await asyncio.gather(*(one(request) for request in requests))

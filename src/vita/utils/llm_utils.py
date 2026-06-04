import json
import re
from typing import Any, Optional

from loguru import logger
from openai import OpenAI

from vita.config import (
    models,
    DEFAULT_MAX_RETRIES,
)
from vita.data_model.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from vita.environment.tool import Tool


_PASSTHROUGH_KEYS = (
    "temperature",
    "max_tokens",
    "max_completion_tokens",
    "reasoning_effort",
    "top_p",
    "stop",
    "seed",
    "extra_body",
    "extra_headers",
)


# Reuse one OpenAI client per (base_url, api_key, max_retries) tuple so the
# underlying httpx connection pool is shared across calls. Otherwise every
# generate() call pays a fresh TCP/TLS handshake under high concurrency.
_CLIENT_CACHE: dict[tuple[str, str, int], OpenAI] = {}


def _get_client(base_url: str, api_key: str, max_retries: int) -> OpenAI:
    key = (base_url, api_key, max_retries)
    client = _CLIENT_CACHE.get(key)
    if client is None:
        client = OpenAI(base_url=base_url, api_key=api_key, max_retries=max_retries)
        _CLIENT_CACHE[key] = client
    return client


# Preserved verbatim from the pre-rewrite llm_utils.py (spec §4.1: these are
# not gateway-specific). External callers (e.g. rewrite_memory) import these
# names; their behaviour must not change across the rewrite.

class DictToObject:
    """
    Convert dictionary to object with attribute access
    Usage:
    response_obj = DictToObject(response)
    print(response_obj.choices[0].message.content)  # Instead of response["choices"][0]["message"]["content"]
    """
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, DictToObject(value))
            elif isinstance(value, list):
                setattr(self, key, [DictToObject(item) if isinstance(item, dict) else item for item in value])
            else:
                setattr(self, key, value)

    def to_dict(self):
        """Convert object back to dictionary"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, DictToObject):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [item.to_dict() if isinstance(item, DictToObject) else item for item in value]
            else:
                result[key] = value
        return result


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from content.

    Some models (e.g. MiniMax-M2.7) embed chain-of-thought directly inside the
    assistant `content` using <think> tags. Downstream consumers (agents,
    user simulators, evaluators) only want the final answer — stripping here
    prevents the reasoning text from being fed back into later turns.
    Separate reasoning fields (`reasoning_content` etc.) are untouched; they
    stay in `raw_data` for inspection but never reach `AssistantMessage.content`.
    """
    if not text or "<think>" not in text:
        return text
    result = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Also drop a stray unterminated opener (truncation / streaming artifacts).
    result = re.sub(r"<think>.*$", "", result, flags=re.DOTALL).strip()
    return result


def get_response_cost(usage: dict, model: str) -> float:
    num_prompt_token = usage.get("prompt_tokens", 0)
    num_completion_token = usage.get("completion_tokens", 0)
    prompt_price = models.get(model, {}).get("cost_1m_token_dollar", {}).get("prompt_price", 0)
    completion_price = models.get(model, {}).get("cost_1m_token_dollar", {}).get("completion_price", 0)
    if prompt_price and completion_price:
        return (prompt_price * num_prompt_token + completion_price * num_completion_token) / 1_000_000
    return 0.0


def get_response_usage(response_dict: dict) -> Optional[dict]:
    usage = response_dict.get("usage")
    if usage is None:
        return None
    return {
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
    }


def format_messages(messages: list[Message]) -> list[dict]:
    out = []
    for m in messages:
        if isinstance(m, UserMessage):
            out.append({"role": "user", "content": m.content})
        elif isinstance(m, AssistantMessage):
            tool_calls = None
            if m.is_tool_call():
                tool_calls = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in m.tool_calls
                ]
            entry = {"role": "assistant", "content": m.content}
            if tool_calls:
                entry["tool_calls"] = tool_calls
            out.append(entry)
        elif isinstance(m, ToolMessage):
            out.append({
                "role": "tool",
                "content": m.content,
                "tool_call_id": m.id,
                "name": m.name,
            })
        elif isinstance(m, SystemMessage):
            out.append({"role": "system", "content": m.content})
    return out


def generate(
    model: str,
    messages: list[Message],
    tools: Optional[list[Tool]] = None,
    tool_choice: Optional[str] = None,
    enable_think: bool = False,
    **kwargs: Any,
) -> AssistantMessage:
    """Generate an assistant message via the OpenAI-compatible chat API.

    `enable_think` is only meaningful for OpenAI o-series / gpt-5: when True
    and the yaml entry omits `reasoning_effort`, we leave the SDK default in
    place. The flag is otherwise a no-op; vendor-specific reasoning toggles
    must be expressed in models.yaml itself.
    """
    cfg = dict(models.get(model, {}))
    base_url = cfg.pop("base_url", None)
    api_key = cfg.pop("api_key", None)
    if not base_url or not api_key:
        raise ValueError(
            f"models.yaml entry for '{model}' is missing base_url or api_key."
        )

    client = _get_client(
        base_url=base_url,
        api_key=api_key,
        max_retries=kwargs.get("num_retries", DEFAULT_MAX_RETRIES),
    )

    request: dict[str, Any] = {
        "model": model,
        "messages": format_messages(messages),
    }
    for k in _PASSTHROUGH_KEYS:
        if k in kwargs and kwargs[k] is not None:
            request[k] = kwargs[k]
        elif k in cfg and cfg[k] is not None:
            request[k] = cfg[k]
    if tools:
        request["tools"] = [t.openai_schema for t in tools]
        request["tool_choice"] = tool_choice or "auto"

    try:
        response = client.chat.completions.create(**request)
    except Exception as e:
        logger.error(f"[model={model}] {e}")
        raise

    response_dict = response.model_dump()
    usage = get_response_usage(response_dict)
    cost = get_response_cost(usage or {}, model)

    choice = response_dict["choices"][0]
    msg = choice["message"]
    assert msg["role"] == "assistant"
    content = msg.get("content")

    raw_tool_calls = msg.get("tool_calls") or []
    tool_calls = [
        ToolCall(
            id=tc.get("id"),
            name=tc.get("function", {}).get("name"),
            arguments=json.loads(tc["function"]["arguments"]) if tc.get("function", {}).get("arguments") else {},
        )
        for tc in raw_tool_calls
    ] or None

    return AssistantMessage(
        role="assistant",
        content=content,
        tool_calls=tool_calls,
        cost=cost,
        usage=usage,
        raw_data=response_dict,
    )


def get_cost(messages: list[Message]) -> tuple[float, float] | None:
    """Aggregate (agent_cost, user_cost) across an interaction."""
    agent_cost = 0.0
    user_cost = 0.0
    for m in messages:
        if isinstance(m, ToolMessage):
            continue
        if m.cost is not None:
            if isinstance(m, AssistantMessage):
                agent_cost += m.cost
            elif isinstance(m, UserMessage):
                user_cost += m.cost
        else:
            logger.warning(f"Message {m.role}: {m.content} has no cost")
            return None
    return agent_cost, user_cost

"""OpenAI-compatible model adapter."""

import json
import os
from typing import Any, Dict, List, Protocol, Sequence

from openai import BadRequestError, OpenAI

from coding_agent.config import ModelConfig
from coding_agent.models import ModelResponse, ToolCall


class ModelProtocolError(RuntimeError):
    """The model produced something the control loop cannot execute.

    Recoverable: the loop can describe the problem and let the model try again.
    """


class LLMClient(Protocol):
    def complete(
        self, messages: List[Dict[str, Any]], tools: Sequence[Dict[str, Any]]
    ) -> ModelResponse:
        ...


class OpenAIChatClient:
    def __init__(self, config: ModelConfig) -> None:
        api_key = os.getenv(config.api_key_env)
        base_url = os.getenv(config.base_url_env)
        if not api_key and not base_url:
            raise ValueError(
                "Set {} (and optionally {}) before running the agent".format(
                    config.api_key_env, config.base_url_env
                )
            )
        self.config = config
        self.client = OpenAI(api_key=api_key or "local", base_url=base_url or None)

    def complete(
        self, messages: List[Dict[str, Any]], tools: Sequence[Dict[str, Any]]
    ) -> ModelResponse:
        try:
            completion = self.client.chat.completions.create(
                model=self.config.name,
                messages=messages,
                tools=list(tools),
                tool_choice="auto",
                temperature=self.config.temperature,
            )
        except BadRequestError as error:
            # Providers that validate tool calls server-side reject a malformed
            # call with 400 rather than returning it for the loop to handle.
            raise ModelProtocolError(str(error)) from error
        if not completion.choices:
            raise ModelProtocolError("Model returned no choices")
        message = completion.choices[0].message
        calls: List[ToolCall] = []
        for call in message.tool_calls or []:
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError as error:
                raise ModelProtocolError(
                    "Invalid JSON arguments for {}: {}".format(call.function.name, error)
                ) from error
            if not isinstance(arguments, dict):
                raise ModelProtocolError("Tool arguments must be a JSON object")
            calls.append(
                ToolCall(
                    call_id=call.id,
                    name=call.function.name,
                    arguments=arguments,
                )
            )

        usage = completion.usage
        return ModelResponse(
            content=message.content or "",
            tool_calls=calls,
            input_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            output_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
        )


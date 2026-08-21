"""The deterministic control loop around a non-deterministic model."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from coding_agent.budget import BudgetExceeded, BudgetState
from coding_agent.config import AppConfig
from coding_agent.llm import LLMClient
from coding_agent.models import AgentResult, ModelResponse, ToolCall
from coding_agent.tools import ToolRegistry
from coding_agent.trajectory import TrajectoryWriter
from coding_agent.workspace import Workspace


SYSTEM_PROMPT = """You are a coding agent working in a single repository.

Your job is to diagnose the user's issue, make the smallest correct code change, and verify it.
Use repository tools instead of guessing. Read relevant code before editing.

Rules:
- Never attempt to access paths outside the repository.
- Prefer exact replacement edits over rewriting whole files.
- Inspect the diff after editing.
- run_tests must pass after the final edit.
- Finish only by calling submit with a concise factual summary.
- If a tool rejects an action, adjust the action; do not repeat it unchanged.
"""


class CodingAgent:
    def __init__(
        self,
        llm: LLMClient,
        repository: Path,
        test_command: Sequence[str],
        config: AppConfig,
        trajectory_path: Path,
    ) -> None:
        self.config = config
        self.llm = llm
        self.trajectory = TrajectoryWriter(trajectory_path)
        self.budget = BudgetState(config.budget, config.model)
        workspace = Workspace(
            repository,
            allowed_command_prefixes=config.workspace.allowed_command_prefixes,
            timeout_seconds=config.agent.command_timeout_seconds,
            max_output_chars=config.agent.max_tool_output_chars,
        )
        self.tools = ToolRegistry(
            workspace,
            test_command=test_command,
            require_tests_before_submit=config.agent.require_tests_before_submit,
            max_output_chars=config.agent.max_tool_output_chars,
        )
        self.last_signature: Optional[str] = None
        self.repeated_action_count = 0

    def run(self, issue: str) -> AgentResult:
        issue = issue.strip()
        if not issue:
            raise ValueError("Issue text cannot be empty")

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "Resolve this issue in the repository:\n\n{}".format(issue),
            },
        ]
        self.trajectory.write("run_started", issue=issue)

        while True:
            try:
                self.budget.begin_step()
            except BudgetExceeded as error:
                return self._finish(False, "budget_exhausted", error.reason)

            try:
                response = self.llm.complete(messages, self.tools.schemas)
            except Exception as error:
                return self._finish(False, "model_error", str(error))

            self._record_model_response(response)
            try:
                self.budget.record_usage(response.input_tokens, response.output_tokens)
            except BudgetExceeded as error:
                return self._finish(False, "budget_exhausted", error.reason)

            messages.append(self._assistant_message(response))
            if not response.tool_calls:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Continue using the available tools. Submit only after the "
                            "verification command passes."
                        ),
                    }
                )
                continue

            for call in response.tool_calls:
                if self._is_repeated(call):
                    self.trajectory.write(
                        "loop_detected",
                        tool=call.name,
                        arguments=call.arguments,
                        repeated=self.repeated_action_count,
                    )
                    return self._finish(
                        False,
                        "loop_detected",
                        "Repeated the same action {} times".format(
                            self.repeated_action_count
                        ),
                    )

                self.trajectory.write(
                    "tool_call",
                    tool_call_id=call.call_id,
                    tool=call.name,
                    arguments=call.arguments,
                )
                result = self.tools.execute(call.name, call.arguments)
                self.trajectory.write(
                    "tool_result",
                    tool_call_id=call.call_id,
                    tool=call.name,
                    ok=result.ok,
                    content=result.content,
                    terminal=result.terminal,
                    metadata=result.metadata,
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.call_id,
                        "content": result.content,
                    }
                )

                if result.terminal:
                    return self._finish(
                        True,
                        "submitted",
                        str(result.metadata["summary"]),
                        patch=str(result.metadata["patch"]),
                    )

    def _is_repeated(self, call: ToolCall) -> bool:
        signature = self.tools.call_signature(call.name, call.arguments)
        if signature == self.last_signature:
            self.repeated_action_count += 1
        else:
            self.last_signature = signature
            self.repeated_action_count = 1
        return self.repeated_action_count >= self.config.agent.max_repeated_action

    def _record_model_response(self, response: ModelResponse) -> None:
        self.trajectory.write(
            "model_response",
            content=response.content,
            tool_calls=[
                {
                    "id": call.call_id,
                    "name": call.name,
                    "arguments": call.arguments,
                }
                for call in response.tool_calls
            ],
            usage={
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
            },
        )

    @staticmethod
    def _assistant_message(response: ModelResponse) -> Dict[str, Any]:
        message: Dict[str, Any] = {
            "role": "assistant",
            "content": response.content or None,
        }
        if response.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.call_id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments),
                    },
                }
                for call in response.tool_calls
            ]
        return message

    def _finish(
        self,
        success: bool,
        status: str,
        summary: str,
        patch: str = "",
    ) -> AgentResult:
        result = AgentResult(
            success=success,
            status=status,
            summary=summary,
            patch=patch,
            steps=self.budget.steps,
            input_tokens=self.budget.input_tokens,
            output_tokens=self.budget.output_tokens,
            cost_usd=self.budget.cost_usd,
            trajectory_path=str(self.trajectory.path),
        )
        self.trajectory.write("run_finished", result=result.as_dict())
        return result


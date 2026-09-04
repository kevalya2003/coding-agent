import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pytest

from coding_agent.agent import CodingAgent
from coding_agent.config import (
    AgentConfig,
    AppConfig,
    BudgetConfig,
    ModelConfig,
    WorkspaceConfig,
)
from coding_agent.llm import ModelProtocolError
from coding_agent.models import ModelResponse, ToolCall
from coding_agent.workspace import WorkspaceError

from conftest import RepositoryFactory


class ScriptedLLM:
    def __init__(self, responses: Sequence[Any]) -> None:
        self.responses = list(responses)
        self.calls = 0
        self.last_messages: List[Dict[str, Any]] = []

    def complete(
        self, messages: List[Dict[str, Any]], tools: Sequence[Dict[str, Any]]
    ) -> ModelResponse:
        self.calls += 1
        self.last_messages = list(messages)
        scripted = self.responses.pop(0)
        if isinstance(scripted, Exception):
            raise scripted
        return scripted


def test_agent_edits_verifies_and_submits(
    tmp_path: Path, make_repository: RepositoryFactory
) -> None:
    repository = make_repository()
    responses = [
        ModelResponse(
            "",
            [ToolCall("call-1", "replace_in_file", {
                "path": "value.txt",
                "old_text": "bad",
                "new_text": "good",
            })],
            input_tokens=10,
            output_tokens=5,
        ),
        ModelResponse("", [ToolCall("call-2", "run_tests", {})], 10, 5),
        ModelResponse(
            "",
            [ToolCall("call-3", "submit", {"summary": "Fix the stored value"})],
            10,
            5,
        ),
    ]
    config = AppConfig(
        model=ModelConfig(),
        budget=BudgetConfig(max_steps=5),
        agent=AgentConfig(),
        workspace=WorkspaceConfig(
            allowed_command_prefixes=((sys.executable, "-c"),)
        ),
    )
    trajectory = tmp_path / "trajectory.jsonl"
    agent = CodingAgent(
        llm=ScriptedLLM(responses),
        repository=repository,
        test_command=[
            sys.executable,
            "-c",
            "from pathlib import Path; assert Path('value.txt').read_text() == 'good\\n'",
        ],
        config=config,
        trajectory_path=trajectory,
    )

    result = agent.run("The stored value must be good.")

    assert result.success is True
    assert result.status == "submitted"
    assert "-bad" in result.patch
    assert "+good" in result.patch
    assert trajectory.read_text(encoding="utf-8").count('"type":') >= 7


def test_submission_is_rejected_until_tests_pass_again(
    tmp_path: Path, make_repository: RepositoryFactory
) -> None:
    repository = make_repository()
    verify = [
        sys.executable,
        "-c",
        "from pathlib import Path; assert Path('value.txt').read_text() == 'good\\n'",
    ]
    responses = [
        ModelResponse("", [ToolCall("c1", "replace_in_file", {
            "path": "value.txt", "old_text": "bad", "new_text": "good",
        })], 10, 5),
        ModelResponse("", [ToolCall("c2", "run_tests", {})], 10, 5),
        ModelResponse("", [ToolCall("c3", "write_file", {
            "path": "late.txt", "content": "added after a green run\n",
        })], 10, 5),
        ModelResponse("", [ToolCall("c4", "submit", {"summary": "too early"})], 10, 5),
        ModelResponse("", [ToolCall("c5", "run_tests", {})], 10, 5),
        ModelResponse("", [ToolCall("c6", "submit", {"summary": "verified"})], 10, 5),
    ]
    config = AppConfig(
        budget=BudgetConfig(max_steps=10),
        workspace=WorkspaceConfig(allowed_command_prefixes=((sys.executable, "-c"),)),
    )
    trajectory = tmp_path / "trajectory.jsonl"
    agent = CodingAgent(
        llm=ScriptedLLM(responses),
        repository=repository,
        test_command=verify,
        config=config,
        trajectory_path=trajectory,
    )

    result = agent.run("The stored value must be good.")
    events = trajectory.read_text(encoding="utf-8")

    assert events.count("Submission rejected") == 1
    assert result.success is True
    assert result.summary == "verified"


VERIFY_GOOD = [
    sys.executable,
    "-c",
    "from pathlib import Path; assert Path('value.txt').read_text() == 'good\\n'",
]


def scripted_config(**agent_options: Any) -> AppConfig:
    return AppConfig(
        budget=BudgetConfig(max_steps=10),
        agent=AgentConfig(**agent_options),
        workspace=WorkspaceConfig(allowed_command_prefixes=((sys.executable, "-c"),)),
    )


def test_a_rejected_tool_call_is_described_to_the_model_and_retried(
    tmp_path: Path, make_repository: RepositoryFactory
) -> None:
    repository = make_repository()
    llm = ScriptedLLM([
        ModelProtocolError(
            "parameters for tool read_file did not match schema: "
            "additionalProperties 'line_start' not allowed"
        ),
        ModelResponse("", [ToolCall("c1", "replace_in_file", {
            "path": "value.txt", "old_text": "bad", "new_text": "good",
        })], 10, 5),
        ModelResponse("", [ToolCall("c2", "run_tests", {})], 10, 5),
        ModelResponse("", [ToolCall("c3", "submit", {"summary": "recovered"})], 10, 5),
    ])
    trajectory = tmp_path / "trajectory.jsonl"

    result = CodingAgent(
        llm=llm,
        repository=repository,
        test_command=VERIFY_GOOD,
        config=scripted_config(),
        trajectory_path=trajectory,
    ).run("The stored value must be good.")

    assert result.success is True
    assert '"protocol_error"' in trajectory.read_text(encoding="utf-8")
    retry_prompt = llm.last_messages[2]["content"]
    assert "line_start" in retry_prompt


def test_repeated_rejections_end_the_run_instead_of_looping(
    tmp_path: Path, make_repository: RepositoryFactory
) -> None:
    repository = make_repository()
    llm = ScriptedLLM([ModelProtocolError("malformed tool call")] * 5)

    result = CodingAgent(
        llm=llm,
        repository=repository,
        test_command=VERIFY_GOOD,
        config=scripted_config(max_protocol_errors=3),
        trajectory_path=tmp_path / "trajectory.jsonl",
    ).run("The stored value must be good.")

    assert result.success is False
    assert result.status == "protocol_error"
    assert llm.calls == 3


def test_agent_refuses_a_verification_command_it_could_never_run(
    tmp_path: Path, make_repository: RepositoryFactory
) -> None:
    repository = make_repository()
    config = AppConfig(
        workspace=WorkspaceConfig(allowed_command_prefixes=(("python", "-m", "pytest"),)),
    )
    llm = ScriptedLLM([])

    with pytest.raises(WorkspaceError, match="not allow-listed"):
        CodingAgent(
            llm=llm,
            repository=repository,
            test_command=["make", "test"],
            config=config,
            trajectory_path=tmp_path / "trajectory.jsonl",
        )

    assert llm.calls == 0


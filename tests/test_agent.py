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
from coding_agent.models import ModelResponse, ToolCall
from coding_agent.workspace import WorkspaceError

from conftest import RepositoryFactory


class ScriptedLLM:
    def __init__(self, responses: Sequence[ModelResponse]) -> None:
        self.responses = list(responses)
        self.calls = 0

    def complete(
        self, messages: List[Dict[str, Any]], tools: Sequence[Dict[str, Any]]
    ) -> ModelResponse:
        self.calls += 1
        return self.responses.pop(0)


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


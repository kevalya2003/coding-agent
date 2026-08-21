import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

from coding_agent.agent import CodingAgent
from coding_agent.config import (
    AgentConfig,
    AppConfig,
    BudgetConfig,
    ModelConfig,
    WorkspaceConfig,
)
from coding_agent.models import ModelResponse, ToolCall


class ScriptedLLM:
    def __init__(self, responses: Sequence[ModelResponse]) -> None:
        self.responses = list(responses)

    def complete(
        self, messages: List[Dict[str, Any]], tools: Sequence[Dict[str, Any]]
    ) -> ModelResponse:
        return self.responses.pop(0)


def initialize_repository(path: Path) -> None:
    (path / "value.txt").write_text("bad\n", encoding="utf-8")
    subprocess.run(["git", "init", "--quiet"], cwd=str(path), check=True)
    subprocess.run(["git", "add", "value.txt"], cwd=str(path), check=True)
    environment = os.environ.copy()
    environment.update(
        {
            "GIT_AUTHOR_NAME": "Test",
            "GIT_AUTHOR_EMAIL": "test@example.com",
            "GIT_COMMITTER_NAME": "Test",
            "GIT_COMMITTER_EMAIL": "test@example.com",
        }
    )
    subprocess.run(
        ["git", "commit", "--quiet", "-m", "baseline"],
        cwd=str(path),
        env=environment,
        check=True,
    )


def test_agent_edits_verifies_and_submits(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    initialize_repository(repository)
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


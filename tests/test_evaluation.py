import json
from pathlib import Path

import pytest

from coding_agent.config import AppConfig, WorkspaceConfig
from coding_agent.evaluation import EvaluationTask, Evaluator, load_tasks


def task(task_id: str, test_command: list) -> EvaluationTask:
    return EvaluationTask(
        task_id=task_id,
        repository_url="https://example.invalid/repository.git",
        base_commit="0" * 40,
        issue="Something is broken.",
        test_command=test_command,
    )


def test_unusable_task_stops_the_run_before_any_model_call(tmp_path: Path) -> None:
    config = AppConfig(
        workspace=WorkspaceConfig(allowed_command_prefixes=(("python", "-m", "pytest"),)),
    )
    output = tmp_path / "out"
    evaluator = Evaluator(llm=None, config=config, output_directory=output)

    with pytest.raises(ValueError, match="not allow-listed"):
        evaluator.run(
            [
                task("good", ["python", "-m", "pytest", "-q"]),
                task("bad", ["make", "test"]),
            ]
        )

    assert not (output / "predictions.jsonl").exists()
    assert not (output / "summary.json").exists()


def test_task_files_reject_duplicate_identifiers(tmp_path: Path) -> None:
    path = tmp_path / "tasks.json"
    entry = {
        "task_id": "same",
        "repository_url": "https://example.invalid/repository.git",
        "base_commit": "0" * 40,
        "issue": "Broken.",
        "test_command": ["python", "-m", "pytest"],
    }
    path.write_text(json.dumps([entry, entry]), encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate task_id"):
        load_tasks(path)


def test_task_files_reject_unsafe_identifiers(tmp_path: Path) -> None:
    path = tmp_path / "tasks.json"
    path.write_text(
        json.dumps(
            [
                {
                    "task_id": "../escape",
                    "repository_url": "https://example.invalid/repository.git",
                    "base_commit": "0" * 40,
                    "issue": "Broken.",
                    "test_command": ["python", "-m", "pytest"],
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsafe task_id"):
        load_tasks(path)

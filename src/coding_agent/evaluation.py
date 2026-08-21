"""Reproducible evaluation over repositories pinned to immutable commits."""

import json
import re
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from coding_agent.agent import CodingAgent
from coding_agent.config import AppConfig
from coding_agent.llm import LLMClient


@dataclass(frozen=True)
class EvaluationTask:
    task_id: str
    repository_url: str
    base_commit: str
    issue: str
    test_command: List[str]
    setup_command: Optional[List[str]] = None


def load_tasks(path: Path) -> List[EvaluationTask]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("Task file must contain a non-empty JSON list")
    tasks: List[EvaluationTask] = []
    seen = set()
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("Every task must be an object")
        task_id = str(item["task_id"])
        if task_id in seen:
            raise ValueError("Duplicate task_id: {}".format(task_id))
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", task_id):
            raise ValueError("Unsafe task_id: {}".format(task_id))
        seen.add(task_id)
        test_command = item["test_command"]
        if not isinstance(test_command, list) or not test_command:
            raise ValueError("{}: test_command must be a list".format(task_id))
        setup_command = item.get("setup_command")
        if setup_command is not None and (
            not isinstance(setup_command, list) or not setup_command
        ):
            raise ValueError("{}: setup_command must be a list".format(task_id))
        tasks.append(
            EvaluationTask(
                task_id=task_id,
                repository_url=str(item["repository_url"]),
                base_commit=str(item["base_commit"]),
                issue=str(item["issue"]),
                test_command=[str(token) for token in test_command],
                setup_command=(
                    [str(token) for token in setup_command]
                    if setup_command is not None
                    else None
                ),
            )
        )
    return tasks


class Evaluator:
    def __init__(
        self,
        llm: LLMClient,
        config: AppConfig,
        output_directory: Path,
    ) -> None:
        self.llm = llm
        self.config = config
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        (self.output_directory / "trajectories").mkdir(exist_ok=True)

    def run(self, tasks: Sequence[EvaluationTask]) -> Dict[str, Any]:
        predictions_path = self.output_directory / "predictions.jsonl"
        results: List[Dict[str, Any]] = []
        with predictions_path.open("w", encoding="utf-8") as predictions:
            for task in tasks:
                result = self._run_task(task)
                results.append(result)
                predictions.write(json.dumps(result, ensure_ascii=False))
                predictions.write("\n")
                predictions.flush()

        successes = sum(1 for result in results if result["success"])
        total_cost = sum(float(result["cost_usd"]) for result in results)
        status_counts = Counter(str(result["status"]) for result in results)
        summary: Dict[str, Any] = {
            "tasks": len(results),
            "submitted": successes,
            "submission_rate": successes / len(results) if results else 0.0,
            "input_tokens": sum(int(result["input_tokens"]) for result in results),
            "output_tokens": sum(int(result["output_tokens"]) for result in results),
            "total_cost_usd": round(total_cost, 6),
            "average_cost_usd": round(total_cost / len(results), 6)
            if results
            else 0.0,
            "status_counts": dict(sorted(status_counts.items())),
        }
        (self.output_directory / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        return summary

    def _run_task(self, task: EvaluationTask) -> Dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="coding-agent-") as temp:
            repository = Path(temp) / "repository"
            clone = subprocess.run(
                ["git", "clone", "--quiet", "--no-checkout", task.repository_url, str(repository)],
                capture_output=True,
                text=True,
                errors="replace",
                check=False,
            )
            if clone.returncode != 0:
                return self._infrastructure_failure(
                    task, "clone_failed", clone.stderr.strip()
                )
            checkout = subprocess.run(
                ["git", "checkout", "--quiet", "--detach", task.base_commit],
                cwd=str(repository),
                capture_output=True,
                text=True,
                errors="replace",
                check=False,
            )
            if checkout.returncode != 0:
                return self._infrastructure_failure(
                    task, "checkout_failed", checkout.stderr.strip()
                )
            if task.setup_command:
                setup = subprocess.run(
                    task.setup_command,
                    cwd=str(repository),
                    capture_output=True,
                    text=True,
                    errors="replace",
                    check=False,
                )
                if setup.returncode != 0:
                    return self._infrastructure_failure(
                        task,
                        "setup_failed",
                        "{}\n{}".format(setup.stdout, setup.stderr).strip(),
                    )

            trajectory = (
                self.output_directory / "trajectories" / "{}.jsonl".format(task.task_id)
            )
            agent = CodingAgent(
                llm=self.llm,
                repository=repository,
                test_command=task.test_command,
                config=self.config,
                trajectory_path=trajectory,
            )
            result = agent.run(task.issue).as_dict()
            result["task_id"] = task.task_id
            result["base_commit"] = task.base_commit
            result["repository_url"] = task.repository_url
            result["model_name_or_path"] = self.config.model.name
            result["model_patch"] = result["patch"]
            return result

    @staticmethod
    def _infrastructure_failure(
        task: EvaluationTask, status: str, summary: str
    ) -> Dict[str, Any]:
        return {
            "task_id": task.task_id,
            "base_commit": task.base_commit,
            "repository_url": task.repository_url,
            "success": False,
            "status": status,
            "summary": summary,
            "patch": "",
            "model_patch": "",
            "steps": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "trajectory_path": None,
        }


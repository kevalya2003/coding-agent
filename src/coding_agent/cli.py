"""Command-line interface for local runs and fixed task-set evaluation."""

import argparse
import json
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence

from dotenv import load_dotenv

from coding_agent.agent import CodingAgent
from coding_agent.config import load_config
from coding_agent.evaluation import Evaluator, load_tasks
from coding_agent.llm import OpenAIChatClient


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "baseline.yaml"


def _command(value: str) -> List[str]:
    arguments = shlex.split(value, posix=True)
    if not arguments:
        raise argparse.ArgumentTypeError("Command cannot be empty")
    return arguments


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="coding-agent",
        description="Run a budgeted, test-gated coding agent.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="YAML configuration file",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    run_parser = subparsers.add_parser("run", help="Run against one local repository")
    run_parser.add_argument("--repo", type=Path, required=True)
    issue_group = run_parser.add_mutually_exclusive_group(required=True)
    issue_group.add_argument("--issue")
    issue_group.add_argument("--issue-file", type=Path)
    run_parser.add_argument(
        "--test-command",
        type=_command,
        required=True,
        help='Quoted command, for example "python -m pytest -q"',
    )
    run_parser.add_argument("--trajectory", type=Path)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Run a pinned JSON task set"
    )
    evaluate_parser.add_argument("--tasks", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    load_dotenv()
    arguments = _parser().parse_args(argv)
    config = load_config(arguments.config)
    llm = OpenAIChatClient(config.model)

    if arguments.subcommand == "run":
        issue = (
            arguments.issue
            if arguments.issue is not None
            else arguments.issue_file.read_text(encoding="utf-8")
        )
        trajectory = arguments.trajectory
        if trajectory is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            trajectory = Path("runs") / "run_{}.jsonl".format(stamp)
        agent = CodingAgent(
            llm=llm,
            repository=arguments.repo,
            test_command=arguments.test_command,
            config=config,
            trajectory_path=trajectory,
        )
        result = agent.run(issue)
        print(json.dumps(result.as_dict(), indent=2))
        return 0 if result.success else 1

    tasks = load_tasks(arguments.tasks)
    summary = Evaluator(llm, config, arguments.output).run(tasks)
    print(json.dumps(summary, indent=2))
    return 0 if summary["submitted"] == summary["tasks"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


"""Git fixtures shared by the workspace, agent and evaluation tests."""

import os
import subprocess
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Sequence

import pytest

RepositoryFactory = Callable[..., Path]
PatchApplier = Callable[[str, Path], int]


def _git_environment() -> Dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "GIT_AUTHOR_NAME": "Test",
            "GIT_AUTHOR_EMAIL": "test@example.com",
            "GIT_COMMITTER_NAME": "Test",
            "GIT_COMMITTER_EMAIL": "test@example.com",
        }
    )
    return environment


def run_git(arguments: Sequence[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git"] + [str(token) for token in arguments],
        cwd=str(cwd),
        env=_git_environment(),
        capture_output=True,
        check=False,
    )


def write_exact(path: Path, content: str) -> None:
    """Write without newline translation so tests can assert on exact bytes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(content)


@pytest.fixture
def make_repository(tmp_path: Path) -> RepositoryFactory:
    def factory(
        name: str = "repository", files: Optional[Mapping[str, str]] = None
    ) -> Path:
        path = tmp_path / name
        path.mkdir(parents=True, exist_ok=True)
        for relative, content in (files or {"value.txt": "bad\n"}).items():
            write_exact(path / relative, content)
        run_git(["init", "--quiet"], path)
        run_git(["config", "core.autocrlf", "false"], path)
        run_git(["add", "-A"], path)
        run_git(["commit", "--quiet", "-m", "baseline"], path)
        return path

    return factory


@pytest.fixture
def apply_patch(tmp_path: Path) -> PatchApplier:
    counter = {"n": 0}

    def apply(patch: str, repository: Path) -> int:
        counter["n"] += 1
        patch_path = tmp_path / "patch-{}.diff".format(counter["n"])
        write_exact(patch_path, patch)
        return run_git(["apply", str(patch_path)], repository).returncode

    return apply

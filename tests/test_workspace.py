import sys
from pathlib import Path

import pytest

from coding_agent.workspace import Workspace, WorkspaceError


def workspace(tmp_path: Path) -> Workspace:
    return Workspace(
        tmp_path,
        allowed_command_prefixes=((sys.executable, "-c"), ("git", "status")),
        timeout_seconds=10,
    )


def test_paths_cannot_escape_repository(tmp_path: Path) -> None:
    subject = workspace(tmp_path)

    with pytest.raises(WorkspaceError, match="escapes"):
        subject.write_file("../outside.txt", "not allowed")

    with pytest.raises(WorkspaceError, match="protected"):
        subject.write_file(".git/config", "not allowed")


def test_edit_invalidates_previous_verification(tmp_path: Path) -> None:
    subject = workspace(tmp_path)
    subject.write_file("value.txt", "before")
    subject.run_tests(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; assert Path('value.txt').read_text() == 'before'",
        ]
    )
    assert subject.tests_verified is True

    subject.replace_in_file("value.txt", "before", "after")

    assert subject.tests_verified is False


def test_non_allow_listed_command_is_rejected(tmp_path: Path) -> None:
    subject = workspace(tmp_path)

    with pytest.raises(WorkspaceError, match="not allow-listed"):
        subject.run_command(["definitely-not-approved", "--version"])


import sys
from pathlib import Path

import pytest

from coding_agent.workspace import Workspace, WorkspaceError

from conftest import PatchApplier, RepositoryFactory


def workspace(tmp_path: Path) -> Workspace:
    return Workspace(
        tmp_path,
        allowed_command_prefixes=((sys.executable, "-c"), ("git", "status")),
        timeout_seconds=10,
    )


def tracked_workspace(repository: Path) -> Workspace:
    return Workspace(repository, allowed_command_prefixes=(("git", "status"),))


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


def test_unusable_verification_command_is_rejected_before_running(tmp_path: Path) -> None:
    subject = workspace(tmp_path)

    with pytest.raises(WorkspaceError, match="not allow-listed"):
        subject.ensure_command_allowed(["make", "test"])

    with pytest.raises(WorkspaceError, match="required"):
        subject.ensure_command_allowed([])


def test_new_file_without_trailing_newline_produces_an_appliable_patch(
    make_repository: RepositoryFactory, apply_patch: PatchApplier
) -> None:
    repository = make_repository()
    subject = tracked_workspace(repository)
    subject.write_file("added/module.py", "def f():\n    return 1")

    patch = subject.get_diff()

    assert "\\ No newline at end of file" in patch
    clean = make_repository("clean")
    assert apply_patch(patch, clean) == 0
    assert (clean / "added" / "module.py").read_bytes() == b"def f():\n    return 1"


def test_new_empty_file_does_not_corrupt_the_patch(
    make_repository: RepositoryFactory, apply_patch: PatchApplier
) -> None:
    repository = make_repository()
    subject = tracked_workspace(repository)
    subject.write_file("placeholder.txt", "")
    subject.replace_in_file("value.txt", "bad", "good")

    clean = make_repository("clean")

    assert apply_patch(subject.get_diff(), clean) == 0
    assert (clean / "placeholder.txt").read_bytes() == b""
    assert (clean / "value.txt").read_bytes() == b"good\n"


def test_edits_preserve_crlf_line_endings(
    make_repository: RepositoryFactory, apply_patch: PatchApplier
) -> None:
    files = {"crlf.txt": "alpha\r\nbeta\r\n"}
    repository = make_repository(files=files)
    subject = tracked_workspace(repository)

    subject.replace_in_file("crlf.txt", "beta", "gamma")

    assert (repository / "crlf.txt").read_bytes() == b"alpha\r\ngamma\r\n"

    clean = make_repository("clean", files=files)
    assert apply_patch(subject.get_diff(), clean) == 0
    assert (clean / "crlf.txt").read_bytes() == b"alpha\r\ngamma\r\n"


def test_written_files_keep_the_content_they_were_given(
    make_repository: RepositoryFactory,
) -> None:
    repository = make_repository()
    subject = tracked_workspace(repository)

    subject.write_file("kept.txt", "one\ntwo\n")

    assert (repository / "kept.txt").read_bytes() == b"one\ntwo\n"


def test_truncation_notice_is_the_last_line(tmp_path: Path) -> None:
    subject = workspace(tmp_path)
    for number in range(205):
        (tmp_path / "file{:03d}.txt".format(number)).write_text("x", encoding="utf-8")

    lines = subject.list_files().splitlines()

    assert lines[0] == "file000.txt"
    assert "result limit reached" in lines[-1]


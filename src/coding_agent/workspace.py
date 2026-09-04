"""Filesystem and process boundary for an agent-controlled repository."""

import fnmatch
import os
import subprocess
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


class WorkspaceError(RuntimeError):
    """A rejected or failed workspace operation."""


def command_is_allowed(
    arguments: Sequence[str], allowed_prefixes: Sequence[Sequence[str]]
) -> bool:
    lowered = tuple(str(token).lower() for token in arguments)
    for prefix in allowed_prefixes:
        normalized = tuple(str(token).lower() for token in prefix)
        if lowered[: len(normalized)] == normalized:
            return True
    return False


def describe_prefixes(allowed_prefixes: Sequence[Sequence[str]]) -> str:
    return "; ".join(" ".join(str(token) for token in p) for p in allowed_prefixes)


class Workspace:
    IGNORED_DIRECTORIES = {".git", ".venv", "node_modules", "__pycache__", "dist", "build"}

    def __init__(
        self,
        root: Path,
        allowed_command_prefixes: Sequence[Sequence[str]],
        timeout_seconds: int = 120,
        max_output_chars: int = 20_000,
    ) -> None:
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise WorkspaceError("Repository does not exist: {}".format(self.root))
        self.allowed_command_prefixes: Tuple[Tuple[str, ...], ...] = tuple(
            tuple(token.lower() for token in prefix)
            for prefix in allowed_command_prefixes
        )
        self.timeout_seconds = timeout_seconds
        self.max_output_chars = max_output_chars
        self.tests_verified = False

    def safe_path(self, relative_path: str, must_exist: bool = False) -> Path:
        path = Path(relative_path)
        if path.is_absolute():
            raise WorkspaceError("Absolute paths are not allowed")
        if any(part.lower() == ".git" for part in path.parts):
            raise WorkspaceError("The .git directory is protected")
        candidate = (self.root / path).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as error:
            raise WorkspaceError("Path escapes the repository") from error
        if must_exist and not candidate.exists():
            raise WorkspaceError("Path does not exist: {}".format(relative_path))
        return candidate

    @staticmethod
    def _read_text(path: Path, errors: str = "strict") -> str:
        """Read without newline translation so edits cannot rewrite line endings."""

        with path.open("r", encoding="utf-8", errors=errors, newline="") as handle:
            return handle.read()

    @staticmethod
    def _write_text(path: Path, content: str) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write(content)

    def _iter_files(self, directory: str = ".") -> Iterable[Path]:
        start = self.safe_path(directory, must_exist=True)
        if not start.is_dir():
            raise WorkspaceError("Not a directory: {}".format(directory))
        for current_root, directory_names, file_names in os.walk(str(start)):
            directory_names[:] = [
                name for name in directory_names if name not in self.IGNORED_DIRECTORIES
            ]
            for file_name in file_names:
                yield Path(current_root) / file_name

    def list_files(self, directory: str = ".", pattern: str = "*", limit: int = 200) -> str:
        matches: List[str] = []
        truncated = False
        for path in self._iter_files(directory):
            relative = path.relative_to(self.root).as_posix()
            if fnmatch.fnmatch(relative, pattern) or fnmatch.fnmatch(path.name, pattern):
                matches.append(relative)
                if len(matches) >= limit:
                    truncated = True
                    break
        if not matches:
            return "(no matching files)"
        listing = sorted(matches)
        if truncated:
            listing.append("... result limit reached ...")
        return "\n".join(listing)

    def search_code(
        self, query: str, directory: str = ".", pattern: str = "*", limit: int = 100
    ) -> str:
        if not query:
            raise WorkspaceError("Search query cannot be empty")
        matches: List[str] = []
        for path in self._iter_files(directory):
            relative = path.relative_to(self.root).as_posix()
            if not (fnmatch.fnmatch(relative, pattern) or fnmatch.fnmatch(path.name, pattern)):
                continue
            try:
                with path.open("r", encoding="utf-8", errors="replace") as handle:
                    for line_number, line in enumerate(handle, start=1):
                        if query.lower() in line.lower():
                            matches.append(
                                "{}:{}:{}".format(relative, line_number, line.rstrip())
                            )
                            if len(matches) >= limit:
                                matches.append("... result limit reached ...")
                                return "\n".join(matches)
            except OSError:
                continue
        return "\n".join(matches) or "(no matches)"

    def read_file(self, relative_path: str, start_line: int = 1, end_line: int = 250) -> str:
        if start_line < 1 or end_line < start_line:
            raise WorkspaceError("Invalid line range")
        path = self.safe_path(relative_path, must_exist=True)
        if not path.is_file():
            raise WorkspaceError("Not a file: {}".format(relative_path))
        lines = self._read_text(path, errors="replace").splitlines()
        selected = lines[start_line - 1 : end_line]
        if not selected:
            return "(requested range is empty)"
        return "\n".join(
            "{:>6} | {}".format(number, line)
            for number, line in enumerate(selected, start=start_line)
        )

    def replace_in_file(self, relative_path: str, old_text: str, new_text: str) -> str:
        if not old_text:
            raise WorkspaceError("old_text cannot be empty")
        path = self.safe_path(relative_path, must_exist=True)
        content = self._read_text(path)
        occurrences = content.count(old_text)
        if occurrences != 1:
            raise WorkspaceError(
                "Expected old_text exactly once, found {} occurrence(s)".format(occurrences)
            )
        self._write_text(path, content.replace(old_text, new_text, 1))
        self.tests_verified = False
        return "Updated {}".format(relative_path)

    def write_file(self, relative_path: str, content: str) -> str:
        path = self.safe_path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._write_text(path, content)
        self.tests_verified = False
        return "Wrote {} characters to {}".format(len(content), relative_path)

    def _prefix_allowed(self, arguments: Sequence[str]) -> bool:
        return command_is_allowed(arguments, self.allowed_command_prefixes)

    def ensure_command_allowed(self, arguments: Sequence[str]) -> None:
        """Reject an unusable verification command before any model call is paid for."""

        if not arguments:
            raise WorkspaceError("A verification command is required")
        if not self._prefix_allowed(arguments):
            raise WorkspaceError(
                "Verification command is not allow-listed: {}. "
                "Allowed prefixes: {}".format(
                    " ".join(str(token) for token in arguments),
                    describe_prefixes(self.allowed_command_prefixes),
                )
            )

    def run_command(self, arguments: Sequence[str]) -> str:
        if not arguments or any(
            "\x00" in str(token) or "\n" in str(token) or "\r" in str(token)
            for token in arguments
        ):
            raise WorkspaceError("Command arguments are invalid")
        if not self._prefix_allowed(arguments):
            raise WorkspaceError("Command is not allow-listed: {}".format(list(arguments)))
        return self._execute(arguments)

    def run_tests(self, arguments: Sequence[str]) -> str:
        self.ensure_command_allowed(arguments)
        output, return_code = self._execute_with_code(arguments)
        self.tests_verified = return_code == 0
        return output

    def _execute(self, arguments: Sequence[str]) -> str:
        output, _ = self._execute_with_code(arguments)
        return output

    def _execute_with_code(self, arguments: Sequence[str]) -> Tuple[str, int]:
        try:
            completed = subprocess.run(
                [str(token) for token in arguments],
                cwd=str(self.root),
                shell=False,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=self.timeout_seconds,
                check=False,
            )
        except FileNotFoundError as error:
            raise WorkspaceError("Executable not found: {}".format(arguments[0])) from error
        except subprocess.TimeoutExpired as error:
            raise WorkspaceError(
                "Command timed out after {} seconds".format(self.timeout_seconds)
            ) from error

        combined = "{}\n{}".format(completed.stdout, completed.stderr).strip()
        if len(combined) > self.max_output_chars:
            combined = combined[-self.max_output_chars :]
            combined = "... output truncated to last {} characters ...\n{}".format(
                self.max_output_chars, combined
            )
        return "exit_code={}\n{}".format(completed.returncode, combined), completed.returncode

    def _git(self, arguments: Sequence[str]) -> Tuple[int, str, str]:
        """Run git and decode output ourselves; text mode would strip CR from diffs."""

        completed = subprocess.run(
            ["git"] + [str(token) for token in arguments],
            cwd=str(self.root),
            capture_output=True,
            check=False,
        )
        return (
            completed.returncode,
            completed.stdout.decode("utf-8", "replace"),
            completed.stderr.decode("utf-8", "replace"),
        )

    def get_diff(self) -> str:
        code, stdout, stderr = self._git(
            ["diff", "HEAD", "--no-ext-diff", "--binary"]
        )
        if code not in (0, 128):
            raise WorkspaceError("Unable to read git diff: {}".format(stderr.strip()))
        patch = stdout if code == 0 else ""

        listed_code, listed, _ = self._git(
            ["ls-files", "--others", "--exclude-standard"]
        )
        if listed_code == 0:
            for relative in listed.split("\n"):
                if not relative:
                    continue
                path = self.safe_path(relative, must_exist=True)
                try:
                    text = self._read_text(path)
                except (OSError, UnicodeDecodeError):
                    continue
                patch += self._new_file_patch(relative, text)
        return patch or "(no changes)"

    @staticmethod
    def _new_file_patch(relative: str, text: str) -> str:
        """Render an untracked file as a new-file hunk `git apply` accepts."""

        posix = Path(relative).as_posix()
        header = "diff --git a/{0} b/{0}\nnew file mode 100644\n".format(posix)
        if not text:
            return header

        lines = text.split("\n")
        final_newline = lines[-1] == ""
        if final_newline:
            lines.pop()

        body = "".join("+{}\n".format(line) for line in lines)
        if not final_newline:
            body += "\\ No newline at end of file\n"

        return "{}--- /dev/null\n+++ b/{}\n@@ -0,0 +1,{} @@\n{}".format(
            header, posix, len(lines), body
        )


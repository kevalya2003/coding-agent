"""Tool schemas and dispatch for the coding agent."""

import json
from typing import Any, Dict, List, Sequence

from coding_agent.models import ToolResult
from coding_agent.workspace import Workspace, WorkspaceError


class ToolRegistry:
    def __init__(
        self,
        workspace: Workspace,
        test_command: Sequence[str],
        require_tests_before_submit: bool = True,
        max_output_chars: int = 20_000,
    ) -> None:
        self.workspace = workspace
        self.test_command = list(test_command)
        self.require_tests_before_submit = require_tests_before_submit
        self.max_output_chars = max_output_chars

    @property
    def schemas(self) -> List[Dict[str, Any]]:
        return [
            self._schema(
                "list_files",
                "List repository files. Use a glob-like pattern such as '*.py'.",
                {
                    "directory": {"type": "string", "default": "."},
                    "pattern": {"type": "string", "default": "*"},
                },
            ),
            self._schema(
                "search_code",
                "Search text case-insensitively and return matching file lines.",
                {
                    "query": {"type": "string"},
                    "directory": {"type": "string", "default": "."},
                    "pattern": {"type": "string", "default": "*"},
                },
                required=["query"],
            ),
            self._schema(
                "read_file",
                "Read a bounded line range from a UTF-8 text file.",
                {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer", "minimum": 1, "default": 1},
                    "end_line": {"type": "integer", "minimum": 1, "default": 250},
                },
                required=["path"],
            ),
            self._schema(
                "replace_in_file",
                "Replace text that occurs exactly once. Read the file first.",
                {
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                },
                required=["path", "old_text", "new_text"],
            ),
            self._schema(
                "write_file",
                "Create or fully overwrite a UTF-8 text file inside the repository.",
                {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                required=["path", "content"],
            ),
            self._schema(
                "run_tests",
                "Run the owner-defined verification command. Call after the final edit.",
                {},
            ),
            self._schema("get_diff", "Show the current unified git diff.", {}),
            self._schema(
                "submit",
                "Submit the verified patch and end the task.",
                {"summary": {"type": "string"}},
                required=["summary"],
            ),
        ]

    @staticmethod
    def _schema(
        name: str,
        description: str,
        properties: Dict[str, Any],
        required: Sequence[str] = (),
    ) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": list(required),
                    "additionalProperties": False,
                },
            },
        }

    def execute(self, name: str, arguments: Dict[str, Any]) -> ToolResult:
        try:
            result = self._execute(name, arguments)
        except (WorkspaceError, KeyError, TypeError, ValueError) as error:
            return ToolResult(ok=False, content="Tool error: {}".format(error))
        return result

    def _execute(self, name: str, arguments: Dict[str, Any]) -> ToolResult:
        if name == "list_files":
            content = self.workspace.list_files(
                directory=str(arguments.get("directory", ".")),
                pattern=str(arguments.get("pattern", "*")),
            )
        elif name == "search_code":
            content = self.workspace.search_code(
                query=str(arguments["query"]),
                directory=str(arguments.get("directory", ".")),
                pattern=str(arguments.get("pattern", "*")),
            )
        elif name == "read_file":
            start = int(arguments.get("start_line", 1))
            end = int(arguments.get("end_line", 250))
            if end - start > 400:
                raise ValueError("read_file is limited to 400 lines per call")
            content = self.workspace.read_file(str(arguments["path"]), start, end)
        elif name == "replace_in_file":
            content = self.workspace.replace_in_file(
                str(arguments["path"]),
                str(arguments["old_text"]),
                str(arguments["new_text"]),
            )
        elif name == "write_file":
            content = self.workspace.write_file(
                str(arguments["path"]), str(arguments["content"])
            )
        elif name == "run_tests":
            content = self.workspace.run_tests(self.test_command)
        elif name == "get_diff":
            content = self.workspace.get_diff()
        elif name == "submit":
            if self.require_tests_before_submit and not self.workspace.tests_verified:
                return ToolResult(
                    ok=False,
                    content="Submission rejected: run_tests must pass after the final edit.",
                )
            patch = self.workspace.get_diff()
            if patch == "(no changes)":
                return ToolResult(
                    ok=False,
                    content="Submission rejected: the repository has no changes.",
                )
            summary = str(arguments["summary"]).strip()
            if not summary:
                raise ValueError("summary cannot be empty")
            return ToolResult(
                ok=True,
                content="Patch accepted: {}".format(summary),
                terminal=True,
                metadata={"summary": summary, "patch": patch},
            )
        else:
            return ToolResult(ok=False, content="Unknown tool: {}".format(name))

        return ToolResult(ok=True, content=self._limit(content))

    def _limit(self, content: str) -> str:
        if len(content) <= self.max_output_chars:
            return content
        return "{}\n... tool output truncated ...".format(
            content[: self.max_output_chars]
        )

    @staticmethod
    def call_signature(name: str, arguments: Dict[str, Any]) -> str:
        return "{}:{}".format(name, json.dumps(arguments, sort_keys=True, default=str))


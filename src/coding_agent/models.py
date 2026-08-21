"""Small provider-neutral data models used by the control loop."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    arguments: Dict[str, Any]


@dataclass(frozen=True)
class ModelResponse:
    content: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    content: str
    terminal: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentResult:
    success: bool
    status: str
    summary: str
    patch: str
    steps: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    trajectory_path: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "status": self.status,
            "summary": self.summary,
            "patch": self.patch,
            "steps": self.steps,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "trajectory_path": self.trajectory_path,
        }


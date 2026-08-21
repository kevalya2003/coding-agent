"""Hard local budgets for model-driven loops."""

from dataclasses import dataclass
from typing import Dict

from coding_agent.config import BudgetConfig, ModelConfig


class BudgetExceeded(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass
class BudgetState:
    limits: BudgetConfig
    pricing: ModelConfig
    steps: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    def begin_step(self) -> None:
        if self.steps >= self.limits.max_steps:
            raise BudgetExceeded("max_steps")
        self.steps += 1

    def record_usage(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens += max(0, input_tokens)
        self.output_tokens += max(0, output_tokens)
        self.cost_usd = (
            self.input_tokens * self.pricing.input_cost_per_million
            + self.output_tokens * self.pricing.output_cost_per_million
        ) / 1_000_000

        if self.input_tokens > self.limits.max_input_tokens:
            raise BudgetExceeded("max_input_tokens")
        if self.output_tokens > self.limits.max_output_tokens:
            raise BudgetExceeded("max_output_tokens")
        if self.cost_usd > self.limits.max_cost_usd:
            raise BudgetExceeded("max_cost_usd")

    def snapshot(self) -> Dict[str, float]:
        return {
            "steps": self.steps,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 8),
        }


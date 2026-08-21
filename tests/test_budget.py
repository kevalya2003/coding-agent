import pytest

from coding_agent.budget import BudgetExceeded, BudgetState
from coding_agent.config import BudgetConfig, ModelConfig


def test_budget_tracks_tokens_steps_and_cost() -> None:
    state = BudgetState(
        BudgetConfig(
            max_steps=2,
            max_input_tokens=100,
            max_output_tokens=50,
            max_cost_usd=1.0,
        ),
        ModelConfig(
            input_cost_per_million=2.0,
            output_cost_per_million=4.0,
        ),
    )

    state.begin_step()
    state.record_usage(20, 10)

    assert state.steps == 1
    assert state.input_tokens == 20
    assert state.output_tokens == 10
    assert state.cost_usd == pytest.approx(0.00008)


def test_budget_stops_before_an_extra_model_call() -> None:
    state = BudgetState(BudgetConfig(max_steps=1), ModelConfig())
    state.begin_step()

    with pytest.raises(BudgetExceeded, match="max_steps"):
        state.begin_step()


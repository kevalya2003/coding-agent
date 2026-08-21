from pathlib import Path

import pytest

from coding_agent.config import ConfigError, load_config


def test_config_expands_environment_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MODEL_NAME", raising=False)
    path = tmp_path / "config.yaml"
    path.write_text(
        """
model:
  name: "${MODEL_NAME:-small-test-model}"
budget:
  max_steps: 3
workspace:
  allowed_command_prefixes:
    - ["python", "-m", "pytest"]
""",
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.model.name == "small-test-model"
    assert config.budget.max_steps == 3
    assert config.workspace.allowed_command_prefixes == (("python", "-m", "pytest"),)


def test_config_rejects_invalid_step_budget(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("budget:\n  max_steps: 0\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="max_steps"):
        load_config(path)


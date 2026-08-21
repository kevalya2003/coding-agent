"""Configuration loading with explicit, typed defaults."""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Match, Optional, Tuple

import yaml


class ConfigError(ValueError):
    """Raised when configuration is missing or invalid."""


@dataclass(frozen=True)
class ModelConfig:
    name: str = "gpt-4.1-mini"
    api_key_env: str = "OPENAI_API_KEY"
    base_url_env: str = "OPENAI_BASE_URL"
    temperature: float = 0.0
    input_cost_per_million: float = 0.0
    output_cost_per_million: float = 0.0


@dataclass(frozen=True)
class BudgetConfig:
    max_steps: int = 20
    max_input_tokens: int = 200_000
    max_output_tokens: int = 20_000
    max_cost_usd: float = 1.0


@dataclass(frozen=True)
class AgentConfig:
    max_repeated_action: int = 3
    command_timeout_seconds: int = 120
    max_tool_output_chars: int = 20_000
    require_tests_before_submit: bool = True


@dataclass(frozen=True)
class WorkspaceConfig:
    allowed_command_prefixes: Tuple[Tuple[str, ...], ...] = field(
        default_factory=lambda: (
            ("python", "-m", "pytest"),
            ("pytest",),
            ("python", "-m", "unittest"),
            ("npm", "test"),
            ("npm", "run", "test"),
            ("cargo", "test"),
            ("go", "test"),
        )
    )


@dataclass(frozen=True)
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    workspace: WorkspaceConfig = field(default_factory=WorkspaceConfig)


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: Match[str]) -> str:
        name = match.group(1)
        default = match.group(2)
        current = os.getenv(name)
        if current is not None:
            return current
        if default is not None:
            return default
        raise ConfigError("Environment variable '{}' is required".format(name))

    return _ENV_PATTERN.sub(replace, value)


def _section(data: Dict[str, Any], name: str) -> Dict[str, Any]:
    value = data.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError("'{}' must be a mapping".format(name))
    return value


def load_config(path: Optional[Path] = None) -> AppConfig:
    """Load YAML configuration, falling back to safe defaults."""

    data: Dict[str, Any] = {}
    if path is not None:
        config_path = Path(path)
        if not config_path.is_file():
            raise ConfigError("Config file does not exist: {}".format(config_path))
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ConfigError("Config root must be a mapping")
        data = _expand_env(loaded)

    model_data = _section(data, "model")
    budget_data = _section(data, "budget")
    agent_data = _section(data, "agent")
    workspace_data = _section(data, "workspace")

    prefixes: List[List[str]] = workspace_data.pop(
        "allowed_command_prefixes",
        [list(prefix) for prefix in WorkspaceConfig().allowed_command_prefixes],
    )
    if not isinstance(prefixes, list) or any(
        not isinstance(prefix, list) or not prefix for prefix in prefixes
    ):
        raise ConfigError("workspace.allowed_command_prefixes must be non-empty lists")

    try:
        config = AppConfig(
            model=ModelConfig(**model_data),
            budget=BudgetConfig(**budget_data),
            agent=AgentConfig(**agent_data),
            workspace=WorkspaceConfig(
                allowed_command_prefixes=tuple(
                    tuple(str(token) for token in prefix) for prefix in prefixes
                ),
                **workspace_data
            ),
        )
    except TypeError as error:
        raise ConfigError("Invalid configuration key: {}".format(error)) from error

    if config.budget.max_steps < 1:
        raise ConfigError("budget.max_steps must be at least 1")
    if config.agent.max_repeated_action < 2:
        raise ConfigError("agent.max_repeated_action must be at least 2")
    return config


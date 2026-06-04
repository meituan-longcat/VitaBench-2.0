import os
import re
import yaml
from pathlib import Path

from loguru import logger

_ENV_PATTERN = re.compile(r"^\$\{([A-Z][A-Z0-9_]*)\}$")


def _interpolate(value):
    """Recursively expand `${VAR}` placeholders in YAML-loaded data."""
    if isinstance(value, str):
        m = _ENV_PATTERN.match(value)
        if m:
            var_name = m.group(1)
            try:
                return os.environ[var_name]
            except KeyError:
                raise KeyError(
                    f"models.yaml references ${{{var_name}}} but the "
                    f"environment variable is not set."
                )
        return value
    if isinstance(value, list):
        return [_interpolate(v) for v in value]
    if isinstance(value, dict):
        return {k: _interpolate(v) for k, v in value.items()}
    return value


def _resolve_yaml_path() -> Path:
    """Pick the right yaml: VITA_MODEL_CONFIG_PATH > models.yaml > models.yaml.example."""
    env_override = os.environ.get("VITA_MODEL_CONFIG_PATH")
    if env_override:
        p = Path(env_override)
        if not p.exists():
            raise FileNotFoundError(
                f"VITA_MODEL_CONFIG_PATH={p} does not exist."
            )
        return p

    here = Path(__file__).parent
    user_yaml = here / "models.yaml"
    if user_yaml.exists():
        return user_yaml

    example_yaml = here / "models.yaml.example"
    if example_yaml.exists():
        logger.warning(
            f"{user_yaml} not found; falling back to models.yaml.example. "
            f"Copy it to models.yaml and customise."
        )
        return example_yaml

    raise FileNotFoundError(
        f"Neither {user_yaml} nor {example_yaml} exists; cannot load model config."
    )


def _deep_merge_dict(base_dict: dict, override_dict: dict) -> dict:
    result = base_dict.copy()
    for key, value in override_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge_dict(result[key], value)
        else:
            result[key] = value
    return result


_models_yaml_path = _resolve_yaml_path()

with open(_models_yaml_path, "r") as f:
    models_config_yaml = yaml.safe_load(f)

models_config_yaml = _interpolate(models_config_yaml)

default_model_config = models_config_yaml.get("default", {})
models = {"default": default_model_config}
for model in models_config_yaml.get("models", []):
    model_name = model["name"]
    merged_config = _deep_merge_dict(default_model_config, model)
    merged_config.pop("name", None)
    models[model_name] = merged_config

logger.info(f"Available models: {list(models.keys())}")

# SIMULATION
DEFAULT_MAX_STEPS = 300
DEFAULT_MAX_RETRIES = 3
DEFAULT_MAX_ERRORS = 10
DEFAULT_SEED = 300
DEFAULT_MAX_CONCURRENCY = 1
DEFAULT_NUM_TRIALS = 1
DEFAULT_SAVE_TO = None
DEFAULT_LOG_LEVEL = "DEBUG"
DEFAULT_LANGUAGE = "chinese"
DEFAULT_EVALUATION_TYPE = "trajectory"
DEFAULT_ENABLE_OUTCOME_REWARD = False

# LLM
DEFAULT_AGENT_IMPLEMENTATION = "llm_agent"
DEFAULT_USER_IMPLEMENTATION = "user_simulator"
DEFAULT_LLM_AGENT = "gpt-4.1"
DEFAULT_LLM_USER = "gpt-4.1"
DEFAULT_LLM_EVALUATOR = "gpt-4.1"

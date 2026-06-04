import os
import pytest
import yaml
from pathlib import Path


def _write_yaml(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "models.yaml"
    p.write_text(body)
    return p


def _load_with_path(yaml_path: Path, monkeypatch):
    """Reload vita.config with VITA_MODEL_CONFIG_PATH pointed at yaml_path.

    Uses pytest's monkeypatch so the env var is restored automatically at
    test teardown — avoids leaking state between tests.
    """
    import importlib
    import vita.config
    monkeypatch.setenv("VITA_MODEL_CONFIG_PATH", str(yaml_path))
    importlib.reload(vita.config)
    return vita.config.models


def test_env_interpolation_succeeds(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_TEST_KEY", "sk-fake-12345")
    yaml_path = _write_yaml(tmp_path, """
default:
  base_url: https://example.com/v1
  api_key: ${MY_TEST_KEY}
models:
  - name: foo
    max_tokens: 10
""")
    models = _load_with_path(yaml_path, monkeypatch)
    assert models["foo"]["api_key"] == "sk-fake-12345"
    assert models["default"]["api_key"] == "sk-fake-12345"


def test_missing_env_var_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("DOES_NOT_EXIST", raising=False)
    yaml_path = _write_yaml(tmp_path, """
default:
  base_url: https://example.com/v1
  api_key: ${DOES_NOT_EXIST}
models:
  - name: foo
""")
    with pytest.raises(KeyError, match="DOES_NOT_EXIST"):
        _load_with_path(yaml_path, monkeypatch)


def test_literal_value_passes_through(tmp_path, monkeypatch):
    yaml_path = _write_yaml(tmp_path, """
default:
  base_url: https://example.com/v1
  api_key: literal-key-no-interp
models:
  - name: foo
""")
    models = _load_with_path(yaml_path, monkeypatch)
    assert models["default"]["api_key"] == "literal-key-no-interp"

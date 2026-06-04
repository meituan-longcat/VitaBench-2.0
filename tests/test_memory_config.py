"""Tests for memory.yaml loading and create_memory config passing."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestLoadMemoryConfig:
    """Test _load_memory_config()."""

    def test_loads_default_and_per_memory_configs(self):
        from vita.memory import _load_memory_config
        # Uses the real memory.yaml that was just created
        configs = _load_memory_config()
        assert "default" in configs
        assert configs["default"]["top_k"] == 5
        assert configs["default"]["similarity_threshold"] == 0.0

    def test_rag_merges_with_default(self):
        from vita.memory import _load_memory_config
        configs = _load_memory_config()
        assert "rag" in configs
        assert configs["rag"]["top_k"] == 8
        assert configs["rag"]["similarity_threshold"] == 0.3

    def test_name_key_stripped(self):
        from vita.memory import _load_memory_config
        configs = _load_memory_config()
        for name, config in configs.items():
            assert "name" not in config, f"'name' key leaked into config for {name}"

    def test_missing_file_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VITA_MEMORY_CONFIG_PATH", str(tmp_path / "nonexistent.yaml"))
        from vita.memory import _load_memory_config
        configs = _load_memory_config()
        assert configs == {}


class TestCreateMemoryPassesConfig:
    """Test that create_memory() passes YAML config to constructor."""

    def test_null_memory_receives_top_k(self):
        from vita.memory import create_memory
        mem = create_memory(memory_type="null", language="chinese")
        assert hasattr(mem, "top_k")
        assert isinstance(mem.top_k, int)

    def test_rewrite_memory_receives_defaults(self):
        from vita.memory import create_memory
        mem = create_memory(memory_type="rewrite", language="chinese")
        assert hasattr(mem, "top_k")
        assert hasattr(mem, "similarity_threshold")

    def test_caller_kwargs_override_yaml(self):
        from vita.memory import create_memory
        mem = create_memory(memory_type="null", language="chinese", top_k=99)
        assert mem.top_k == 99

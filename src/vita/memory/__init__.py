"""
Memory module for the personalization domain.

Provides extensible memory backends for storing and retrieving
user preference information across subtasks.

Usage:
    # Built-in memory (string shortcut)
    memory = create_memory(memory_type="rewrite", language="chinese")

    # Custom memory (import path)
    memory = create_memory(memory_class="my_pkg.VectorMemory", language="chinese")
"""

import importlib
import logging
import os
import warnings
from pathlib import Path
from typing import Optional

import yaml

from vita.memory.base import BaseMemory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Memory YAML config loading
# ---------------------------------------------------------------------------

def _deep_merge_dict(base: dict, override: dict) -> dict:
    """Deep merge override into base, returning a new dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def _load_memory_config() -> dict:
    """Load memory configuration from memory.yaml.

    Returns dict[str, dict] keyed by memory name.
    Gracefully returns empty dict if file is missing.
    """
    config_path = Path(__file__).parent.parent / "memory.yaml"
    env_path = os.environ.get("VITA_MEMORY_CONFIG_PATH")
    if env_path:
        config_path = Path(env_path)

    if not config_path.exists():
        return {}

    try:
        with open(config_path, "r") as f:
            raw = yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning(f"Failed to load memory.yaml: {e}")
        return {}

    default_config = raw.get("default", {})
    configs = {"default": default_config}

    for entry in raw.get("memory", []):
        if not isinstance(entry, dict) or "name" not in entry:
            continue
        name = entry["name"]
        # Strip 'name' key — not a constructor param
        entry_config = {k: v for k, v in entry.items() if k != "name"}
        configs[name] = _deep_merge_dict(default_config, entry_config)

    return configs


_memory_configs = _load_memory_config()


# Registry maps short names to dotted import paths (lazy loading).
MEMORY_REGISTRY = {
    "rewrite": "vita.memory.rewrite_memory.RewriteMemory",
    "rag": "vita.memory.rag_memory.RAGMemory",
    "rag_cache": "vita.memory.rag_cache.RAGCacheMemory",
    "null": "vita.memory.null_memory.NullMemory",
    "full_context": "vita.memory.full_context.FullContextMemory",
    "groundtruth": "vita.memory.groundtruth_memory.GroundtruthMemory",
}


def _import_class(dotpath: str):
    """Import a class from a dotted path like 'my_pkg.module.ClassName'.

    Args:
        dotpath: Full dotted path to the class.

    Returns:
        The imported class.

    Raises:
        ImportError: If the module cannot be imported.
        AttributeError: If the class is not found in the module.
    """
    module_path, class_name = dotpath.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def create_memory(
    memory_type: str = None,
    memory_class: str = None,
    language: Optional[str] = None,
    **kwargs,
) -> BaseMemory:
    """Create a memory instance.

    Two usage modes:
        create_memory(memory_type="rewrite")              # built-in shortcut
        create_memory(memory_class="my_pkg.VectorMemory") # custom class

    Args:
        memory_type: Name of a built-in memory type ('rewrite', 'rag', 'null').
        memory_class: Dotted import path to a custom BaseMemory subclass.
        language: Language setting for the memory.
        **kwargs: Additional arguments passed to the memory constructor.

    Returns:
        A BaseMemory instance.

    Raises:
        ValueError: If memory_type is not registered.
        TypeError: If the class does not inherit from BaseMemory.
        ImportError: If memory_class cannot be imported.
    """
    if memory_class and memory_type:
        warnings.warn(
            "Both --memory-class and --memory-type provided; using --memory-class",
            stacklevel=2,
        )

    if memory_class:
        cls = _import_class(memory_class)
    elif memory_type:
        dotpath = MEMORY_REGISTRY.get(memory_type)
        if dotpath is None:
            available = ", ".join(MEMORY_REGISTRY.keys())
            raise ValueError(
                f"Unknown memory type: '{memory_type}'. Available types: {available}"
            )
        cls = _import_class(dotpath)
    else:
        # Default to rewrite
        cls = _import_class(MEMORY_REGISTRY["rewrite"])

    if not issubclass(cls, BaseMemory):
        raise TypeError(
            f"{cls.__name__} must inherit from BaseMemory"
        )

    # Merge YAML config with caller kwargs (caller kwargs take precedence)
    yaml_config = _memory_configs.get(memory_type or "default", {})
    merged_kwargs = {**yaml_config, **kwargs}

    return cls(language=language, **merged_kwargs)

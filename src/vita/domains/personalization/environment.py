"""
Environment setup and task loading for the personalization domain.

The personalization domain reuses cross-domain environments from
delivery, instore, and ota. Memory tools are injected by the
PersonalizationOrchestrator at runtime (not here).
"""

import json
from typing import List, Optional

from loguru import logger

from vita.data_model.personalization_task import PersonalizationTask
from vita.environment.environment import Environment, get_agent_policy
from vita.utils.utils import DOMAIN_DIR


def get_environment(
    db: Optional[dict] = None,
    language: str = None,
) -> Environment:
    """Creates a basic personalization environment.

    Note: The actual per-subtask environment is set up dynamically by the
    PersonalizationOrchestrator based on each subtask's domain. This function
    provides a default environment for registry registration.

    Args:
        db: Optional environment data dictionary.
        language: Language setting.

    Returns:
        A basic Environment with personalization policy.
    """
    if language is not None:
        from vita.utils.schema_utils import set_global_language
        set_global_language(language)

    from vita.prompts import get_prompts
    prompts = get_prompts(language)
    policy = prompts.personalization_agent_system_prompt

    return Environment(
        domain_name="personalization",
        policy=policy,
        tools=None,  # Tools are set up per-subtask by orchestrator
    )


def get_tasks(language: str = None) -> List[PersonalizationTask]:
    """Load personalization tasks from the data directory.

    Args:
        language: Language setting (for future i18n support).

    Returns:
        List of PersonalizationTask objects.
    """
    task_dir = DOMAIN_DIR / "personalization"

    if language == "english":
        task_path = task_dir / "tasks_en.json"
        if not task_path.exists():
            task_path = task_dir / "tasks.json"
    else:
        task_path = task_dir / "tasks.json"

    if not task_path.exists():
        logger.warning(
            f"Personalization task file not found: {task_path}. "
            f"Run scripts/generate_personalization_data.py to generate sample data."
        )
        return []

    with open(task_path, "r") as fp:
        tasks_data = json.load(fp)

    tasks = [PersonalizationTask.model_validate(task) for task in tasks_data]
    logger.info(f"Loaded {len(tasks)} personalization tasks from {task_path}")
    return tasks

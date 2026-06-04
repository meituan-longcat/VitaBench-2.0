"""
GroundtruthMemory: Provides the agent with the ground-truth user preference
from task data (user_scenario.personalized_preference_memory).

This serves as an **upper-bound baseline** — if the agent had perfect
preference knowledge, what's the best reward it can achieve?

The orchestrator sets the current subtask's ground-truth preference via
`set_groundtruth(preference_dict)` before each subtask conversation.
read() formats it as a human-readable string for the system prompt.
update() is a no-op (preferences come from task data, not from interactions).

Usage:
    vita run --domain personalization --memory-type groundtruth
"""

import json
import logging
from typing import Optional, Dict, Any

from vita.memory.base import BaseMemory

logger = logging.getLogger(__name__)


class GroundtruthMemory(BaseMemory):
    """Ground-truth preference memory — oracle baseline.

    read() returns the ground-truth preference for the current subtask.
    update() is a no-op.
    No @is_tool methods: the preference is injected via system prompt only.
    """

    def __init__(self, language: str = None, **kwargs):
        super().__init__(language=language, **kwargs)
        self._preference: Dict[str, Any] = {}
        self._historical_chat: list = []
        self._historical_behavior: Dict[str, Any] = {}

    def set_groundtruth(
        self,
        preference: Dict[str, Any],
        historical_chat: list = None,
        historical_behavior: Dict[str, Any] = None,
    ):
        """Set the ground-truth preference for the current subtask.

        Called by the orchestrator before each subtask conversation.

        Args:
            preference: The personalized_preference_memory dict from
                        subtask.user_scenario, typically containing
                        'current' (preference tags by category) and
                        optionally 'preference_tag_change_history'.
            historical_chat: Recent chat snippets from subtask.historical_chat.
            historical_behavior: Behavioral statistics from subtask.historical_behavior.
        """
        self._preference = preference or {}
        self._historical_chat = historical_chat or []
        self._historical_behavior = historical_behavior or {}
        logger.info(
            f"Groundtruth preference set: "
            f"{len(self._preference)} keys, "
            f"{len(self._historical_chat)} chat items, "
            f"{len(self._historical_behavior)} behavior keys"
        )

    def read(self, query: str = None) -> str:
        """Return the ground-truth preference, historical chat, and behavior formatted as readable text."""
        sections = []

        # --- Preference tags ---
        if self._preference:
            current = self._preference
            if isinstance(current, dict) and "current" in current:
                current = current["current"]

            pref_lines = []
            if isinstance(current, dict):
                for category, tags in current.items():
                    if isinstance(tags, list) and tags:
                        pref_lines.append(f"【{category}】")
                        for tag in tags:
                            pref_lines.append(f"  - {tag}")
                    elif isinstance(tags, str) and tags:
                        pref_lines.append(f"【{category}】{tags}")
            else:
                pref_lines.append(json.dumps(current, ensure_ascii=False, indent=2))

            if pref_lines:
                sections.append("### 用户偏好标签\n" + "\n".join(pref_lines))

        # --- Historical chat ---
        if self._historical_chat:
            chat_lines = ["### 用户历史对话片段"]
            for item in self._historical_chat:
                chat_lines.append(f"  - {item}")
            sections.append("\n".join(chat_lines))

        # --- Historical behavior ---
        if self._historical_behavior:
            behavior_lines = ["### 用户历史行为数据"]
            for key, val in self._historical_behavior.items():
                if not val:
                    continue
                if isinstance(val, list):
                    if val:
                        behavior_lines.append(f"【{key}】{', '.join(str(v) for v in val)}")
                elif isinstance(val, dict):
                    behavior_lines.append(f"【{key}】")
                    for scene, scene_data in val.items():
                        if not scene_data:
                            continue
                        if isinstance(scene_data, dict):
                            relevant = {k: v for k, v in scene_data.items() if v}
                            if relevant:
                                behavior_lines.append(f"  {scene}:")
                                for k, v in relevant.items():
                                    if isinstance(v, list):
                                        v = ', '.join(str(x) for x in v)
                                    behavior_lines.append(f"    - {k}: {v}")
                        elif scene_data:
                            behavior_lines.append(f"  {scene}: {scene_data}")
                else:
                    behavior_lines.append(f"【{key}】{val}")
            if len(behavior_lines) > 1:
                sections.append("\n".join(behavior_lines))

        if not sections:
            return "No user preference information available yet."

        return "\n\n".join(sections)

    def update(
        self,
        new_interactions: list,
        llm: Optional[str] = None,
        llm_args: Optional[dict] = None,
        **kwargs,
    ) -> str:
        """No-op. Ground-truth preference comes from task data, not interactions."""
        return self.read()

    def reset(self):
        """Clear the stored preference."""
        self._preference = {}

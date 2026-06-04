"""
PersonalizationAgent: An LLM agent with preference memory awareness.

Extends LLMAgent to:
1. Inject memory content into the system prompt
2. Process interactions to update memory between subtasks
3. Support memory-related tools for active memory querying
"""

from copy import deepcopy
from typing import List, Optional

from loguru import logger

from vita.agent.llm_agent import LLMAgent, LLMAgentState
from vita.data_model.message import SystemMessage, Message
from vita.data_model.personalization_task import Interaction
from vita.environment.tool import Tool
from vita.memory.base import BaseMemory
from vita.prompts import get_prompts
from vita.utils.utils import get_now, get_weekday


class PersonalizationAgent(LLMAgent):
    """An LLM agent with user preference memory.

    The agent's system prompt is augmented with the current state of the
    user's preference memory. Memory is updated between subtasks.
    """

    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        memory: BaseMemory,
        user_profile: Optional[dict] = None,
        llm: Optional[str] = None,
        llm_args: Optional[dict] = None,
        time: Optional[str] = None,
        enable_think: bool = False,
        language: str = None,
    ):
        """Initialize the PersonalizationAgent.

        Args:
            tools: Available tools for the agent.
            domain_policy: Domain policy string (system prompt template).
            memory: Memory backend instance for user preferences.
            user_profile: User account info (address, demographics, etc.).
            llm: LLM model name.
            llm_args: LLM arguments.
            time: Current simulation time.
            enable_think: Whether to enable thinking mode.
            language: Language setting.
        """
        super().__init__(
            tools=tools,
            domain_policy=domain_policy,
            llm=llm,
            llm_args=llm_args,
            time=time,
            enable_think=enable_think,
            language=language,
        )
        self.memory = memory
        self.user_profile = user_profile or {}
        self._current_instruction: Optional[str] = None

    def set_current_instruction(self, instruction: str):
        """Set the current subtask instruction for memory retrieval.

        Called by the orchestrator before each subtask. The instruction
        is used as the retrieve query in memory.read().
        """
        self._current_instruction = instruction

    def _format_user_profile(self) -> str:
        """Format user_profile dict into a readable block for the system prompt."""
        if not self.user_profile:
            return ""
        lines = []
        for key, value in self.user_profile.items():
            if isinstance(value, list):
                value = "、".join(str(v) for v in value)
            lines.append(f"- {key}: {value}")
        return "\n".join(lines)

    @property
    def system_prompt(self) -> str:
        """Generate system prompt with user profile and memory content injected."""
        base_prompt = self.domain_policy
        if self.time is not None:
            base_prompt = base_prompt.format(time=self.time)
        else:
            base_prompt = base_prompt.format(time=get_now("%Y-%m-%d %H:%M:%S"))

        # Inject user account info (always available, like a real service)
        profile_section = ""
        profile_text = self._format_user_profile()
        if profile_text:
            profile_section = (
                "\n\n## 当前用户基础信息\n"
                "以下是该用户的账户注册信息，可直接使用：\n"
                f"{profile_text}"
            )

        memory_content = self.memory.read(query=self._current_instruction)
        return (
            f"{base_prompt}"
            f"{profile_section}\n\n"
            f"## User Preference Memory\n"
            f"Below is the accumulated knowledge about this user's preferences "
            f"from past interactions. Use this information to better serve the user, "
            f"especially when their requests are vague or ambiguous:\n\n"
            f"{memory_content}"
        )

    def process_interactions(self, interactions: list):
        """Update memory with new interactions. Called between subtasks.

        Args:
            interactions: List of new interactions to process.
        """
        if not interactions:
            logger.debug("No new interactions to process for memory update")
            return
        logger.info(
            f"Updating memory with {len(interactions)} new interactions"
        )
        self.memory.update(
            new_interactions=interactions,
            llm=self.llm,
            llm_args=self.llm_args,
        )
        logger.info(f"Memory updated. Current memory:\n{self.memory.read()[:200]}...")

    def reset_state(self, message_history: Optional[list[Message]] = None) -> LLMAgentState:
        """Reset agent state for a new subtask while preserving memory.

        This re-initializes the conversation state (system prompt + message history)
        but keeps the memory intact across subtasks.

        Args:
            message_history: Optional message history for the new subtask.

        Returns:
            Fresh LLMAgentState with updated system prompt.
        """
        return self.get_init_state(message_history=message_history)

    def update_tools(self, tools: List[Tool]):
        """Update the available tools (e.g., when switching subtask domains).

        Args:
            tools: New list of tools.
        """
        self.tools = tools

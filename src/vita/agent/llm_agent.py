from copy import deepcopy
from typing import List, Optional

from loguru import logger
from pydantic import BaseModel

from vita.agent.base import (
    LocalAgent,
    ValidAgentInputMessage,
    is_valid_agent_history_message,
)
from vita.data_model.message import (
    APICompatibleMessage,
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
)
from vita.environment.tool import Tool
from vita.utils.llm_utils import generate
from vita.utils.utils import get_now, get_weekday


class LLMAgentState(BaseModel):
    """The state of the agent."""

    system_messages: list[SystemMessage]
    messages: list[APICompatibleMessage]


class LLMAgent(LocalAgent[LLMAgentState]):
    """
    An LLM agent that can be used to solve a task.
    """

    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        llm: Optional[str] = None,
        llm_args: Optional[dict] = None,
        time=None,
        enable_think: bool = False,
        language: str = None
    ):
        """
        Initialize the LLMAgent.
        """
        super().__init__(tools=tools, domain_policy=domain_policy)
        self.llm = llm
        self.llm_args = deepcopy(llm_args) if llm_args is not None else {}
        self.time = time + " " + get_weekday(time, language)
        self.enable_think = enable_think

    @property
    def system_prompt(self) -> str:
        if self.time is not None:
            return self.domain_policy.format(
                time=self.time
            )
        return self.domain_policy.format(
            time=get_now("%Y-%m-%d %H:%M:%S")
        )

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> LLMAgentState:
        """Get the initial state of the agent.

        Args:
            message_history: The message history of the conversation.

        Returns:
            The initial state of the agent.
        """
        if message_history is None:
            message_history = []
        assert all(is_valid_agent_history_message(m) for m in message_history), (
            "Message history must contain only AssistantMessage, UserMessage, or ToolMessage to Agent."
        )

        return LLMAgentState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=message_history,
        )

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: LLMAgentState
    ) -> tuple[AssistantMessage, LLMAgentState]:
        """
        Respond to a user or tool message.
        """
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        messages = state.system_messages + state.messages

        assistant_message = generate(
            model=self.llm,
            tools=self.tools,
            messages=messages,
            enable_think=self.enable_think,
            **self.llm_args,
        )
        if assistant_message is not None:
            state.messages.append(assistant_message)

        return assistant_message, state


    def set_seed(self, seed: int):
        """Set the seed for the LLM."""
        if self.llm is None:
            raise ValueError("LLM is not set")
        cur_seed = self.llm_args.get("seed", None)
        if cur_seed is not None:
            logger.warning(f"Seed is already set to {cur_seed}, resetting it to {seed}")
        self.llm_args["seed"] = seed

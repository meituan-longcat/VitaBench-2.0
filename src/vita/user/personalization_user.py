"""
PersonalizationUser: A user simulator that dispatches subtasks sequentially.

Key behavior:
- Sends subtask instructions one at a time to the agent
- During subtask execution, responds to clarification questions via LLM
  (but responses are intentionally vague since the agent should use memory)
- When agent signals completion (###STOP###), marks subtask as done and moves to next
"""

from typing import List, Optional, Tuple

from loguru import logger

from vita.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    ToolCall,
    UserMessage,
)
from vita.data_model.personalization_task import SubTask
from vita.user.base import (
    STOP,
    TRANSFER,
    OUT_OF_SCOPE,
    BaseUser,
    UserState,
    ValidUserInputMessage,
    is_valid_user_history_message,
)
from vita.utils.llm_utils import generate
from vita.prompts import get_prompts


class PersonalizationUser(BaseUser):
    """A user simulator that dispatches pre-defined subtasks sequentially.

    Uses LLM for mid-subtask clarification responses. The user is intentionally
    vague in responses to force the agent to rely on preference memory.
    """

    def __init__(
        self,
        subtasks: List[SubTask],
        persona: str,
        instructions: Optional[str] = None,
        llm: Optional[str] = None,
        llm_args: Optional[dict] = None,
        language: str = None,
    ):
        """Initialize the PersonalizationUser.

        Args:
            subtasks: List of subtasks to dispatch.
            persona: User persona description.
            instructions: Overall instructions (optional).
            llm: LLM model name for generating responses.
            llm_args: LLM arguments.
            language: Language setting.
        """
        super().__init__(instructions=instructions, llm=llm, llm_args=llm_args)
        self.subtasks = subtasks
        self.persona = persona
        self.language = language
        self.current_subtask_idx = 0
        self.subtask_completed = False
        self._subtask_started = False

    @property
    def current_subtask(self) -> Optional[SubTask]:
        """Get the current subtask."""
        if self.current_subtask_idx < len(self.subtasks):
            return self.subtasks[self.current_subtask_idx]
        return None

    @property
    def all_subtasks_done(self) -> bool:
        """Check if all subtasks have been completed."""
        return self.current_subtask_idx >= len(self.subtasks)

    def get_next_subtask_instruction(self) -> Optional[str]:
        """Get the instruction for the current subtask.

        Returns:
            The instruction string, or None if all subtasks are done.
        """
        subtask = self.current_subtask
        if subtask is None:
            return None
        return subtask.instruction

    def advance_to_next_subtask(self):
        """Mark current subtask as done and advance to the next one."""
        self.current_subtask_idx += 1
        self.subtask_completed = False
        self._subtask_started = False
        logger.info(
            f"Advanced to subtask {self.current_subtask_idx}/{len(self.subtasks)}"
        )

    def mark_subtask_completed(self):
        """Mark the current subtask as completed."""
        self.subtask_completed = True
        logger.info(
            f"Subtask {self.current_subtask_idx} marked as completed"
        )

    @property
    def system_prompt(self) -> str:
        """Generate system prompt for the user simulator."""
        prompts = get_prompts(self.language)
        base_prompt = prompts.personalization_user_system_prompt

        subtask = self.current_subtask
        instruction = subtask.instruction if subtask is not None else "All tasks completed."

        # Inject proactive section only for proactive subtasks that carry a non-empty user_intention
        proactive_section = ""
        if (
            subtask is not None
            and "proactive" in subtask.skill_tested
            and subtask.user_intention
        ):
            section_template = prompts.personalization_user_proactive_section
            proactive_section = section_template.format(
                proactive_info=subtask.user_intention
            ) + "\n"

        return base_prompt.format(
            persona=self.persona,
            instruction=instruction,
            proactive_section=proactive_section,
        )

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> UserState:
        """Get the initial state of the user simulator.

        Args:
            message_history: Optional message history.

        Returns:
            Initial UserState.
        """
        if message_history is None:
            message_history = []
        assert all(is_valid_user_history_message(m) for m in message_history), (
            "Invalid user message history."
        )

        user_state = UserState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=message_history,
        )
        return user_state

    @classmethod
    def is_stop(cls, message: UserMessage) -> bool:
        """Check if the message is a stop message.

        Args:
            message: The user message to check.

        Returns:
            True if the message indicates stopping.
        """
        if message.is_tool_call():
            return False
        assert message.content is not None
        return (
            STOP in message.content
            or TRANSFER in message.content
            or OUT_OF_SCOPE in message.content
        )

    def generate_next_message(
        self, message: ValidUserInputMessage, state: UserState
    ) -> Tuple[UserMessage, UserState]:
        """Generate the next user message.

        If starting a new subtask, sends the subtask instruction.
        If the agent asks a clarification question, responds via LLM (vaguely).

        Args:
            message: The incoming message (from agent or tool).
            state: Current user state.

        Returns:
            Tuple of (user message, updated state).
        """
        # Add the incoming message to state
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        # If not started yet, send the subtask instruction
        if not self._subtask_started:
            self._subtask_started = True
            instruction = self.get_next_subtask_instruction()
            if instruction is None:
                # All subtasks done
                user_message = UserMessage(
                    role="user",
                    content=f"All my tasks are done, thank you! {STOP}",
                )
                state.messages.append(user_message)
                return user_message, state

            user_message = UserMessage(
                role="user",
                content=instruction,
            )
            state.messages.append(user_message)
            return user_message, state

        # Use LLM to generate a response (intentionally vague)
        # Update system messages to reflect current subtask context
        state.system_messages = [
            SystemMessage(role="system", content=self.system_prompt)
        ]
        messages = state.system_messages + state.flip_roles()

        assistant_message = generate(
            model=self.llm,
            messages=messages,
            tools=None,
            **self.llm_args,
        )

        user_response = assistant_message.content
        logger.debug(f"PersonalizationUser response: {user_response}")

        user_message = UserMessage(
            role="user",
            content=user_response,
            cost=assistant_message.cost,
            usage=assistant_message.usage,
            raw_data=assistant_message.raw_data,
        )

        if assistant_message.tool_calls is not None:
            user_message.tool_calls = []
            for tool_call in assistant_message.tool_calls:
                user_message.tool_calls.append(
                    ToolCall(
                        id=tool_call.id,
                        name=tool_call.name,
                        arguments=tool_call.arguments,
                        requestor="user",
                    )
                )

        state.messages.append(user_message)
        return user_message, state

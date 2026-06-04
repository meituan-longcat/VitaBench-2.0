"""
Data models for the personalization domain.

A PersonalizationTask represents a single user with multiple sequential subtasks.
Each subtask is a separate service request (delivery, instore, ota) that the agent
must handle while leveraging a preference memory built from prior interactions.
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field
from typing_extensions import Annotated

from vita.data_model.message import Message
from vita.data_model.tasks import EvaluationCriteria, UserScenario


class Interaction(BaseModel):
    """A single user interaction (search, order, conversation, etc.)"""

    type: Annotated[
        str,
        Field(
            description="Type of interaction: 'search', 'order', 'conversation', 'browse', 'review', etc."
        ),
    ]
    timestamp: Annotated[
        str,
        Field(description="Timestamp of the interaction in format YYYY-MM-DD HH:MM:SS"),
    ]
    content: Annotated[
        Any,
        Field(description="Flexible content of the interaction (dict, str, list, etc.)"),
    ]


class SubTask(BaseModel):
    """A single subtask within a personalization task."""

    subtask_id: Annotated[
        str,
        Field(description="Unique identifier for the subtask"),
    ]
    domain: Annotated[
        str,
        Field(
            description="Domain of this subtask: 'delivery', 'instore', or 'ota'"
        ),
    ]
    instruction: Annotated[
        str,
        Field(
            description="Vague instruction for the subtask, e.g. '帮我点个外卖'"
        ),
    ]
    environment: Annotated[
        dict,
        Field(description="Domain-specific environment data for this subtask"),
    ]
    user_scenario: Annotated[
        Optional[UserScenario],
        Field(
            description="User scenario for this subtask (user profile, etc.)",
            default=None,
        ),
    ]
    evaluation_criteria: Annotated[
        Optional[EvaluationCriteria],
        Field(
            description="Evaluation criteria for this subtask",
            default=None,
        ),
    ]
    interactions: Annotated[
        List[Any],
        Field(
            description="Interaction records carried by this subtask, used to update memory. "
            "Accepts both Interaction objects ({type, timestamp, content}) and "
            "init_gen format ({date, behavior, dialogue}).",
            default_factory=list,
        ),
    ]
    skill_tested: Annotated[
        List[str],
        Field(
            description="Skills being tested in this subtask, e.g. ['proactive'], ['update']. "
            "Empty list means regular personalization test.",
            default_factory=list,
        ),
    ]
    user_intention: Annotated[
        Optional[str],
        Field(
            description="Hidden user intention. Disclosed only when the agent proactively "
            "asks a directly related question. Used when skill_tested contains 'proactive'.",
            default="",
        ),
    ]
    message_history: Annotated[
        Optional[List[Message]],
        Field(
            description="Optional message history to seed the subtask conversation",
            default=None,
        ),
    ]


class PersonalizationTask(BaseModel):
    """A personalization task = one user with multiple sequential subtasks.

    The agent maintains a preference memory that is:
    1. Bootstrapped from the first subtask's interactions
    2. Updated with each subsequent subtask's interactions
    3. Injected into agent system prompt AND queryable via tools
    """

    id: Annotated[
        str,
        Field(description="User-level task ID"),
    ]
    user_id: Annotated[
        str,
        Field(description="User identifier"),
    ]
    user_profile: Annotated[
        Dict[str, Any],
        Field(description="User demographic and preference info"),
    ]
    subtasks: Annotated[
        List[SubTask],
        Field(description="Ordered list of subtasks to execute sequentially"),
    ]

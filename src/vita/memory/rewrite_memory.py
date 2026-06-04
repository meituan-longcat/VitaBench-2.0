"""
RewriteMemory: A simple text-based memory backend.

On each update:
1. Takes current memory text + new interactions
2. Calls an LLM to rewrite/merge into an updated preference text
3. Replaces old memory with the new text
"""

import json
from typing import Optional

from loguru import logger

from vita.memory.base import BaseMemory
from vita.environment.toolkit import ToolType, is_tool
from vita.prompts import get_prompts
from vita.utils.llm_utils import generate


class RewriteMemory(BaseMemory):
    """Simple text rewrite memory.

    Maintains a single text blob that is rewritten by an LLM each time
    new interactions are added. Good for small-to-medium preference sets.
    """

    def __init__(self, language: str = None, max_tokens: int = 2048, **kwargs):
        super().__init__(language=language, **kwargs)
        self.max_tokens = max_tokens
        self._memory_text: str = ""

    def read(self, query: str = None) -> str:
        """Return the current memory content. Query parameter is ignored (rewrite is always a full summary)."""
        if not self._memory_text:
            return "No user preference information available yet."
        return self._memory_text

    def update(
        self,
        new_interactions: list,
        llm: Optional[str] = None,
        llm_args: Optional[dict] = None,
        **kwargs,
    ) -> str:
        """Update memory by asking an LLM to merge current memory with new interactions.

        Args:
            new_interactions: New interactions to incorporate.
            llm: LLM model name.
            llm_args: LLM arguments.

        Returns:
            Updated memory text.
        """
        if not new_interactions:
            return self._memory_text

        # Format new interactions as text
        interactions_text = self._format_interactions(new_interactions)

        if llm is None:
            # Without an LLM, just append the interactions to memory
            if self._memory_text:
                self._memory_text = (
                    f"{self._memory_text}\n\n"
                    f"Additional observations from recent interactions:\n{interactions_text}"
                )
            else:
                self._memory_text = (
                    f"User preference observations from interactions:\n{interactions_text}"
                )
            return self._memory_text

        # Use LLM to rewrite memory
        from vita.data_model.message import SystemMessage, UserMessage

        prompts = get_prompts(self.language)
        memory_update_prompt = prompts.memory_update_prompt

        user_content = memory_update_prompt.format(
            current_memory=self._memory_text or "Empty (no preferences recorded yet)",
            new_interactions=interactions_text,
        )
        # Tell the model its output budget. Without this, the model has no
        # idea how long a summary is acceptable, and generate(max_tokens=...)
        # silently truncates anything that runs past it. Matches the hard cap
        # passed to the gateway below.
        if self.language == "english":
            user_content += f"\n\n(Please keep the updated summary within about {self.max_tokens} tokens.)"
        else:
            user_content += f"\n\n(请将总结控制在约 {self.max_tokens} 个 token 以内。)"

        messages = [
            SystemMessage(
                role="system",
                content="You are a preference memory manager. Your job is to maintain an accurate, concise summary of user preferences based on their interaction history.",
            ),
            UserMessage(role="user", content=user_content),
        ]

        if llm_args is None:
            llm_args = {}
        # Caller's llm_args may already contain max_tokens (e.g. agent
        # llm_args = models[model] from yaml), which collides with the
        # explicit max_tokens kwarg below. Strip it so rewrite's own
        # self.max_tokens wins without a TypeError.
        llm_args = {
            k: v for k, v in llm_args.items()
            if k not in ("max_tokens", "max_completion_tokens")
        }

        try:
            # max_tokens passed as kwarg wins over the per-model yaml default
            # via _PASSTHROUGH_KEYS in llm_utils.generate().
            response = generate(
                model=llm,
                messages=messages,
                max_tokens=self.max_tokens,
                **llm_args,
            )
            if response is not None and response.content:
                self._memory_text = response.content.strip()
            else:
                logger.warning(
                    "LLM returned empty/None response for memory update, falling back to append"
                )
                if self._memory_text:
                    self._memory_text = (
                        f"{self._memory_text}\n\n"
                        f"Additional observations:\n{interactions_text}"
                    )
                else:
                    self._memory_text = (
                        f"User preference observations:\n{interactions_text}"
                    )
        except Exception as e:
            logger.error(f"Error during LLM memory update: {e}. Falling back to append.")
            if self._memory_text:
                self._memory_text = (
                    f"{self._memory_text}\n\n"
                    f"Additional observations:\n{interactions_text}"
                )
            else:
                self._memory_text = (
                    f"User preference observations:\n{interactions_text}"
                )

        return self._memory_text

    # ── Tools: auto-discovered by framework via @is_tool ──

    @is_tool(ToolType.READ)
    def read_preference_memory(self) -> str:
        """读取用户偏好记忆，获取关于用户偏好的完整知识"""
        return self.read()

    @is_tool(ToolType.READ)
    def query_preference_memory(self, query: str) -> str:
        """根据具体问题查询用户偏好记忆"""
        # RewriteMemory always returns full text since it's already a concise summary
        if not self._memory_text:
            return "No user preference information available."
        return self._memory_text

    def reset(self):
        """Reset memory to empty state."""
        self._memory_text = ""

    @staticmethod
    def _format_interactions(interactions: list) -> str:
        """Format a list of interactions as human-readable text.

        Supports two formats:
        1. Interaction objects: {type, timestamp, content}
        2. init_gen format: {date, behavior: [...], dialogue: [...]}
        """
        lines = []
        type_labels = {
            "order": "订单",
            "search": "搜索",
            "conversation": "对话",
            "browse": "浏览",
            "review": "评价",
            "high_freq_browse": "高频浏览",
            "rate": "评分",
            "comment": "评论",
            "complaint": "投诉",
            "add_to_cart": "加购",
            "favorite": "收藏",
        }

        for interaction in interactions:
            # Handle dict (could be either format)
            if isinstance(interaction, dict):
                # init_gen format: {date, behavior, dialogue}
                if "date" in interaction and ("behavior" in interaction or "dialogue" in interaction):
                    date = interaction.get("date", "")
                    # Format behaviors
                    for beh in interaction.get("behavior", []):
                        if isinstance(beh, dict):
                            btype = beh.get("behavior_type", "unknown")
                            label = type_labels.get(btype, btype)
                            content = beh.get("content", {})
                            content_str = (
                                json.dumps(content, ensure_ascii=False)
                                if isinstance(content, (dict, list))
                                else str(content)
                            )
                            lines.append(f"- [{date}] [{label}] {content_str}")
                    # Format dialogue
                    dialogue = interaction.get("dialogue", [])
                    if dialogue:
                        dialogue_str = json.dumps(dialogue, ensure_ascii=False)
                        lines.append(f"- [{date}] [对话] {dialogue_str}")
                # Interaction-like format: {type, timestamp, content}
                elif "type" in interaction and "content" in interaction:
                    itype = interaction.get("type", "unknown")
                    label = type_labels.get(itype, itype)
                    ts = interaction.get("timestamp", "")
                    content = interaction.get("content", "")
                    content_str = (
                        json.dumps(content, ensure_ascii=False, indent=2)
                        if isinstance(content, (dict, list))
                        else str(content)
                    )
                    lines.append(f"- [{ts}] [{label}] {content_str}")
                else:
                    # Unknown dict format, dump as-is
                    lines.append(f"- {json.dumps(interaction, ensure_ascii=False)}")
            # Handle Interaction pydantic model objects
            elif hasattr(interaction, "type") and hasattr(interaction, "content"):
                content_str = (
                    json.dumps(interaction.content, ensure_ascii=False, indent=2)
                    if isinstance(interaction.content, (dict, list))
                    else str(interaction.content)
                )
                label = type_labels.get(interaction.type, interaction.type)
                lines.append(
                    f"- [{interaction.timestamp}] [{label}] {content_str}"
                )
            else:
                lines.append(f"- {str(interaction)}")

        return "\n".join(lines)

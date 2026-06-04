"""Test that instruction flows from orchestrator to memory.read(query)."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestInstructionFlow:
    """Test that PersonalizationAgent correctly passes instruction to memory."""

    def test_set_current_instruction(self):
        """Test that set_current_instruction stores the instruction."""
        with patch("vita.agent.personalization_agent.LLMAgent.__init__", return_value=None):
            from vita.agent.personalization_agent import PersonalizationAgent
            agent = PersonalizationAgent.__new__(PersonalizationAgent)
            agent.memory = MagicMock()
            agent._current_instruction = None
            agent.set_current_instruction("帮我点一份外卖")
            assert agent._current_instruction == "帮我点一份外卖"

    def test_system_prompt_passes_instruction_to_read(self):
        """Test that system_prompt calls memory.read(query=instruction)."""
        with patch("vita.agent.personalization_agent.LLMAgent.__init__", return_value=None):
            from vita.agent.personalization_agent import PersonalizationAgent
            agent = PersonalizationAgent.__new__(PersonalizationAgent)
            agent.memory = MagicMock()
            agent.memory.read.return_value = "mocked memory"
            agent.domain_policy = "Policy for {time}"
            agent.time = "2024-01-01 12:00"
            agent.language = "chinese"
            agent._current_instruction = "帮我订一个酒店"

            _ = agent.system_prompt

            agent.memory.read.assert_called_once_with(query="帮我订一个酒店")

    def test_system_prompt_no_instruction_passes_none(self):
        """Test backward compat: no instruction means read(query=None)."""
        with patch("vita.agent.personalization_agent.LLMAgent.__init__", return_value=None):
            from vita.agent.personalization_agent import PersonalizationAgent
            agent = PersonalizationAgent.__new__(PersonalizationAgent)
            agent.memory = MagicMock()
            agent.memory.read.return_value = "mocked memory"
            agent.domain_policy = "Policy for {time}"
            agent.time = "2024-01-01 12:00"
            agent.language = "chinese"
            agent._current_instruction = None

            _ = agent.system_prompt

            agent.memory.read.assert_called_once_with(query=None)

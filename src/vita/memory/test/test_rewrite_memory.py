"""
Tests for RewriteMemory.

Covers:
- read() / update() / reset() core interface
- _format_interactions() for both init_gen and Interaction model formats
- LLM-free fallback (append mode) when llm=None
- Multi-turn memory accumulation
"""

import pytest

from vita.memory.rewrite_memory import RewriteMemory


class TestRewriteMemoryBasic:
    """Basic lifecycle: empty -> update -> read -> reset."""

    def test_initial_read_returns_placeholder(self):
        mem = RewriteMemory(language="chinese")
        assert "No user preference" in mem.read()

    def test_reset_clears_memory(self, delivery_interactions_initgen):
        mem = RewriteMemory(language="chinese")
        mem.update(delivery_interactions_initgen)
        assert mem.read() != ""
        mem.reset()
        assert "No user preference" in mem.read()

    def test_read_query_param_ignored(self, delivery_interactions_initgen):
        """RewriteMemory 的 read(query=...) 应始终返回完整文本."""
        mem = RewriteMemory(language="chinese")
        mem.update(delivery_interactions_initgen)
        full = mem.read()
        queried = mem.read(query="回锅肉")
        assert full == queried


class TestRewriteMemoryUpdateNoLLM:
    """Without LLM: update() should append formatted interactions."""

    def test_update_delivery_initgen(self, delivery_interactions_initgen):
        mem = RewriteMemory(language="chinese")
        result = mem.update(delivery_interactions_initgen)
        # 应包含搜索/订单关键词
        assert "红烧肉盖饭" in result
        assert "川味盖饭王" in result
        assert "回锅肉" in result

    def test_update_instore_initgen(self, instore_interactions_initgen):
        mem = RewriteMemory(language="chinese")
        result = mem.update(instore_interactions_initgen)
        assert "火锅" in result or "蜀九香" in result
        assert "鸳鸯锅" in result or "微辣" in result

    def test_update_ota_initgen(self, ota_interactions_initgen):
        mem = RewriteMemory(language="chinese")
        result = mem.update(ota_interactions_initgen)
        assert "青年旅舍" in result or "梦之旅" in result

    def test_update_model_format(self, delivery_interactions_model):
        mem = RewriteMemory(language="chinese")
        result = mem.update(delivery_interactions_model)
        assert "红烧肉盖饭" in result
        assert "份量超足" in result or "review" in result.lower() or "评价" in result

    def test_update_empty_interactions(self, empty_interactions):
        mem = RewriteMemory(language="chinese")
        result = mem.update(empty_interactions)
        assert result == ""

    def test_update_accumulates(self, delivery_interactions_initgen, ota_interactions_initgen):
        """多轮 update 应该累积所有信息."""
        mem = RewriteMemory(language="chinese")
        mem.update(delivery_interactions_initgen)
        mem.update(ota_interactions_initgen)
        result = mem.read()
        assert "红烧肉" in result or "盖饭" in result
        assert "青年旅舍" in result or "梦之旅" in result


class TestRewriteMemoryMultiTurn:
    """Simulate the orchestrator calling update() across multiple subtask turns."""

    def test_three_turn_evolution(self, multi_turn_interactions):
        mem = RewriteMemory(language="chinese")
        for turn_interactions in multi_turn_interactions:
            mem.update(turn_interactions)

        final = mem.read()
        # Turn 1 & 2: 川菜偏好
        assert "川味盖饭王" in final or "红烧肉" in final or "回锅肉" in final
        # Turn 3: 酒旅偏好
        assert "青旅" in final or "梦之旅" in final or "经济" in final


class TestFormatInteractions:
    """Test _format_interactions static method with various input shapes."""

    def test_initgen_with_behavior_and_dialogue(self):
        interactions = [
            {
                "date": "2023-03-10",
                "behavior": [
                    {"behavior_type": "order", "content": {"merchant_name": "测试店铺"}},
                ],
                "dialogue": [
                    {"role": "user", "content": "帮我点单"},
                ],
            }
        ]
        result = RewriteMemory._format_interactions(interactions)
        assert "2023-03-10" in result
        assert "订单" in result
        assert "测试店铺" in result
        assert "对话" in result

    def test_initgen_behavior_only(self, single_behavior_no_dialogue):
        result = RewriteMemory._format_interactions(single_behavior_no_dialogue)
        assert "搜索" in result
        assert "塔斯汀" in result

    def test_initgen_dialogue_only(self, dialogue_only_no_behavior):
        result = RewriteMemory._format_interactions(dialogue_only_no_behavior)
        assert "对话" in result
        assert "优惠活动" in result

    def test_model_format_with_type_labels(self):
        interactions = [
            {"type": "order", "timestamp": "2023-03-10 12:00:00", "content": {"items": ["红烧肉"]}},
            {"type": "search", "timestamp": "2023-03-10 11:00:00", "content": {"keyword": "盖饭"}},
            {"type": "browse", "timestamp": "2023-03-10 10:00:00", "content": "看看有什么好吃的"},
            {"type": "review", "timestamp": "2023-03-10 13:00:00", "content": {"rating": 5, "text": "好吃"}},
            {"type": "complaint", "timestamp": "2023-03-10 14:00:00", "content": "配送太慢"},
            {"type": "add_to_cart", "timestamp": "2023-03-10 15:00:00", "content": {"item": "薯条"}},
            {"type": "favorite", "timestamp": "2023-03-10 16:00:00", "content": {"store": "华莱士"}},
        ]
        result = RewriteMemory._format_interactions(interactions)
        assert "订单" in result
        assert "搜索" in result
        assert "浏览" in result
        assert "评价" in result
        assert "投诉" in result
        assert "加购" in result
        assert "收藏" in result

    def test_unknown_dict_format(self):
        interactions = [{"random_key": "random_value"}]
        result = RewriteMemory._format_interactions(interactions)
        assert "random_value" in result

    def test_pharmacy_interactions(self, pharmacy_interactions_initgen):
        result = RewriteMemory._format_interactions(pharmacy_interactions_initgen)
        assert "氯雷他定" in result or "药" in result
        assert "康佳大药房" in result


class TestRewriteMemoryTools:
    """Test @is_tool methods exposed by RewriteMemory."""

    def test_read_preference_memory_tool(self, delivery_interactions_initgen):
        mem = RewriteMemory(language="chinese")
        mem.update(delivery_interactions_initgen)
        result = mem.read_preference_memory()
        assert result == mem.read()

    def test_query_preference_memory_tool(self, delivery_interactions_initgen):
        mem = RewriteMemory(language="chinese")
        mem.update(delivery_interactions_initgen)
        result = mem.query_preference_memory("红烧肉")
        assert result == mem.read()

    def test_tools_registered(self):
        mem = RewriteMemory(language="chinese")
        tool_names = set(mem.tools.keys())
        assert "read_preference_memory" in tool_names
        assert "query_preference_memory" in tool_names

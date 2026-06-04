"""Tool schema definitions for the personalization domain.

This file contains the descriptions and mappings for memory tools
used in the personalization domain.
"""

from typing import Dict, Any

# Tool descriptions for memory tools - Chinese version
TOOL_DESCRIPTIONS_ZH = {
    "read_preference_memory": {
        "description": "读取用户偏好记忆，获取关于用户偏好的完整知识",
        "preconditions": "需要了解用户的偏好信息时使用",
        "postconditions": "返回当前用户偏好记忆内容",
        "args": {},
        "returns": "用户偏好记忆内容字符串",
        "tool_type": "READ"
    },
    "query_preference_memory": {
        "description": "根据具体问题查询用户偏好记忆",
        "preconditions": "需要查询用户特定偏好信息时使用",
        "postconditions": "返回与查询相关的偏好信息",
        "args": {
            "query": "关于用户偏好的自然语言问题"
        },
        "returns": "与查询相关的偏好信息",
        "tool_type": "READ"
    },
}

# Tool descriptions for memory tools - English version
TOOL_DESCRIPTIONS_EN = {
    "read_preference_memory": {
        "description": "Read user preference memory to get full knowledge about user preferences",
        "preconditions": "Use when you need to understand user preference information",
        "postconditions": "Returns the current user preference memory content",
        "args": {},
        "returns": "User preference memory content string",
        "tool_type": "READ"
    },
    "query_preference_memory": {
        "description": "Query user preference memory with a specific question",
        "preconditions": "Use when you need to query specific user preference information",
        "postconditions": "Returns preference information relevant to the query",
        "args": {
            "query": "Natural language question about user preferences"
        },
        "returns": "Preference information relevant to the query",
        "tool_type": "READ"
    },
}

TOOL_DESCRIPTIONS = {
    "chinese": TOOL_DESCRIPTIONS_ZH,
    "english": TOOL_DESCRIPTIONS_EN,
}

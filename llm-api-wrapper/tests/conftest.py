"""测试公共 fixtures"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# 确保 src 在 import 路径中
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_messages() -> list[dict[str, str]]:
    """标准测试消息列表"""
    return [
        {"role": "system", "content": "你是一个有帮助的助手。"},
        {"role": "user", "content": "你好，请介绍一下自己。"},
        {"role": "assistant", "content": "你好！我是一个 AI 助手。"},
        {"role": "user", "content": "今天天气怎么样？"},
    ]


@pytest.fixture
def long_messages() -> list[dict[str, str]]:
    """用于测试上下文裁剪的长消息列表"""
    msgs = [{"role": "system", "content": "你是一个有帮助的助手。"}]
    for i in range(20):
        msgs.append({"role": "user", "content": f"这是第 {i+1} 条用户消息，" * 50})
        msgs.append({"role": "assistant", "content": f"这是第 {i+1} 条回复，" * 50})
    return msgs


@pytest.fixture
def mock_chat_response_data() -> dict:
    """模拟 API 返回的原始 JSON 数据"""
    return {
        "id": "chatcmpl-test-123",
        "model": "qwen-turbo",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "你好！我是通义千问。",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": 10,
            "total_tokens": 30,
        },
    }


@pytest.fixture
def mock_structured_response_data() -> dict:
    """模拟结构化 JSON 输出的 API 响应"""
    return {
        "id": "chatcmpl-test-456",
        "model": "qwen-turbo",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": '{"name": "张三", "age": 25, "city": "北京"}',
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 50,
            "completion_tokens": 20,
            "total_tokens": 70,
        },
    }

"""上下文裁剪模块测试"""

import pytest

from src.utils.context_trimmer import trim_context
from src.utils.tokens import count_messages_tokens


class TestTrimContext:
    """trim_context 单元测试"""

    def test_no_trim_needed(self, sample_messages):
        """token 未超限时原样返回"""
        max_tokens = 10000
        result = trim_context(sample_messages, max_tokens)
        assert result == sample_messages

    def test_empty_messages(self):
        result = trim_context([], 100)
        assert result == []

    def test_preserves_system_message(self, long_messages):
        """裁剪后应保留 system 消息"""
        max_tokens = 200
        result = trim_context(long_messages, max_tokens)
        system_msgs = [m for m in result if m["role"] == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0]["content"] == "你是一个有帮助的助手。"

    def test_preserves_recent_messages(self, long_messages):
        """裁剪后应保留最近的对话"""
        max_tokens = 500
        result = trim_context(long_messages, max_tokens, preserve_recent=2)
        # 最后两条（非 system）应被保留
        non_system = [m for m in result if m["role"] != "system"]
        if non_system:
            # 最后一条应该是原始列表的最后一条
            assert non_system[-1] == long_messages[-1]

    def test_result_within_limit(self, long_messages):
        """裁剪结果不应超过 max_tokens"""
        max_tokens = 300
        result = trim_context(long_messages, max_tokens)
        actual_tokens = count_messages_tokens(result)
        assert actual_tokens <= max_tokens

    def test_oldest_first_removes_earliest(self):
        """oldest_first 策略应从最早的对话开始移除"""
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "msg1 " * 50},
            {"role": "assistant", "content": "reply1 " * 50},
            {"role": "user", "content": "msg2 " * 50},
            {"role": "assistant", "content": "reply2 " * 50},
            {"role": "user", "content": "msg3"},
            {"role": "assistant", "content": "reply3"},
        ]
        # 设置一个让前面的消息需要被移除的限制
        full_tokens = count_messages_tokens(messages)
        max_tokens = full_tokens - 100  # 强制裁剪一些

        result = trim_context(messages, max_tokens, strategy="oldest_first")
        # 最后两条短消息应被保留
        non_system = [m for m in result if m["role"] != "system"]
        last_msgs = non_system[-2:] if len(non_system) >= 2 else non_system
        assert any("msg3" in m["content"] for m in last_msgs)

    def test_single_message_within_limit(self):
        """单条消息不超限时不裁剪"""
        messages = [{"role": "user", "content": "短文本"}]
        result = trim_context(messages, 1000)
        assert result == messages

    def test_custom_preserve_recent(self):
        """自定义 preserve_recent 参数"""
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "old " * 100},
            {"role": "user", "content": "mid " * 100},
            {"role": "user", "content": "new1"},
            {"role": "user", "content": "new2"},
            {"role": "user", "content": "new3"},
        ]
        max_tokens = 200
        result = trim_context(messages, max_tokens, preserve_recent=3)
        non_system = [m for m in result if m["role"] != "system"]
        # 应保留最近 3 条
        assert any("new3" in m["content"] for m in non_system)

"""Token 计数模块测试"""

import pytest

from src.utils.tokens import count_tokens, count_messages_tokens


class TestCountTokens:
    """count_tokens 单元测试"""

    def test_empty_string(self):
        assert count_tokens("") == 0

    def test_english_text(self):
        result = count_tokens("Hello, world!")
        assert result > 0
        assert isinstance(result, int)

    def test_chinese_text(self):
        result = count_tokens("你好世界")
        assert result > 0

    def test_mixed_text(self):
        result = count_tokens("Hello 你好 World 世界")
        assert result > 0

    def test_long_text_proportional(self):
        short = count_tokens("abc")
        long = count_tokens("abc " * 100)
        assert long > short

    def test_consistent_results(self):
        """相同输入应返回相同结果"""
        text = "测试文本 test text 123"
        assert count_tokens(text) == count_tokens(text)


class TestCountMessagesTokens:
    """count_messages_tokens 单元测试"""

    def test_empty_messages(self):
        assert count_messages_tokens([]) == 0

    def test_single_message(self):
        messages = [{"role": "user", "content": "你好"}]
        result = count_messages_tokens(messages)
        # 至少包含内容 token + 消息开销 (4) + 回复引导 (3)
        assert result >= 7

    def test_multiple_messages(self, sample_messages):
        result = count_messages_tokens(sample_messages)
        assert result > 0
        # 多条消息应大于单条
        single = count_messages_tokens([sample_messages[0]])
        assert result > single

    def test_message_overhead(self):
        """每条消息应有固定开销"""
        msg1 = [{"role": "user", "content": "a"}]
        msg2 = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
        ]
        diff = count_messages_tokens(msg2) - count_messages_tokens(msg1)
        # 第二条消息至少贡献 4 (overhead) + content tokens
        assert diff >= 4

    def test_missing_content(self):
        """content 为空时不应报错"""
        messages = [{"role": "user", "content": ""}]
        result = count_messages_tokens(messages)
        assert result > 0  # 至少有开销 token

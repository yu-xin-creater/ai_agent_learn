"""Token 计数工具

基于 tiktoken (cl100k_base) 提供 token 计数能力。
注意：与 Qwen/DeepSeek 实际 tokenizer 存在 ≤5% 误差，作为预估使用。
"""

from __future__ import annotations

import tiktoken

# 使用 cl100k_base 编码（GPT-4 / ChatGPT 系列通用）
_ENCODING_NAME = "cl100k_base"
_encoding: tiktoken.Encoding | None = None


def _get_encoding() -> tiktoken.Encoding:
    """懒加载 tiktoken 编码器"""
    global _encoding
    if _encoding is None:
        _encoding = tiktoken.get_encoding(_ENCODING_NAME)
    return _encoding


def count_tokens(text: str) -> int:
    """计算单段文本的 token 数量

    Args:
        text: 待计数的文本

    Returns:
        token 数量
    """
    if not text:
        return 0
    enc = _get_encoding()
    return len(enc.encode(text))


def count_messages_tokens(messages: list[dict[str, str]]) -> int:
    """计算消息列表的总 token 数量

    采用 OpenAI Chat 格式的计算规则：
    - 每条消息固定开销 4 tokens (role + content 标记)
    - 最终追加 3 tokens (assistant reply priming)

    Args:
        messages: 消息列表，每条含 role 和 content 字段

    Returns:
        总 token 数量
    """
    if not messages:
        return 0

    enc = _get_encoding()
    total = 0

    for msg in messages:
        # 每条消息的固定开销：<|im_start|>{role}\n ... <|im_end|>\n
        total += 4
        content = msg.get("content", "")
        if content:
            total += len(enc.encode(content))
        role = msg.get("role", "")
        if role:
            total += len(enc.encode(role))

    # 最后追加 assistant 回复引导 token
    total += 3
    return total

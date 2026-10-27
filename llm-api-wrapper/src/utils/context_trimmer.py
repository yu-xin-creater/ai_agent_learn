"""上下文裁剪工具

当消息列表的 token 数超过模型窗口时，按策略裁剪以适配限制。
"""

from __future__ import annotations

from typing import Literal

from src.utils.tokens import count_messages_tokens


TrimStrategy = Literal["oldest_first"]


def trim_context(
    messages: list[dict[str, str]],
    max_tokens: int,
    strategy: TrimStrategy = "oldest_first",
    *,
    preserve_recent: int = 2,
) -> list[dict[str, str]]:
    """裁剪消息上下文使其 token 数不超过限制

    Args:
        messages: 原始消息列表
        max_tokens: 允许的最大 token 数
        strategy: 裁剪策略，目前支持 "oldest_first"
        preserve_recent: 保留最近 N 轮对话（不含 system 消息）

    Returns:
        裁剪后的消息列表

    策略说明:
        oldest_first: 保留 system 消息 + 最近 preserve_recent 轮对话，
                      从最早的非 system 消息开始逐条移除，直到满足 token 限制。
    """
    if not messages:
        return []

    # 快速检查：如果已经满足限制，原样返回
    current_tokens = count_messages_tokens(messages)
    if current_tokens <= max_tokens:
        return list(messages)

    if strategy == "oldest_first":
        return _trim_oldest_first(messages, max_tokens, preserve_recent)

    # 未知策略回退为 oldest_first
    return _trim_oldest_first(messages, max_tokens, preserve_recent)


def _trim_oldest_first(
    messages: list[dict[str, str]],
    max_tokens: int,
    preserve_recent: int,
) -> list[dict[str, str]]:
    """oldest_first 裁剪策略

    1. 分离 system 消息和对话消息
    2. 对话消息中保留最近 preserve_recent 条
    3. 从最早的可裁剪消息开始逐条移除
    """
    # 分离 system 消息
    system_msgs = [m for m in messages if m.get("role") == "system"]
    conversation_msgs = [m for m in messages if m.get("role") != "system"]

    if not conversation_msgs:
        return list(messages)

    # 保护最近 N 条对话
    protected_count = min(preserve_recent, len(conversation_msgs))
    protected_msgs = conversation_msgs[-protected_count:] if protected_count > 0 else []
    trimmable_msgs = conversation_msgs[:-protected_count] if protected_count > 0 else conversation_msgs

    # 逐条从最早的开始移除
    while trimmable_msgs:
        candidate = system_msgs + trimmable_msgs + protected_msgs
        tokens = count_messages_tokens(candidate)
        if tokens <= max_tokens:
            return candidate
        # 移除最早的一条
        trimmable_msgs.pop(0)

    # 全部可裁剪消息移除后，检查剩余是否满足
    result = system_msgs + protected_msgs
    tokens = count_messages_tokens(result)
    if tokens <= max_tokens:
        return result

    # 极端情况：连 protected 也要裁剪
    while protected_msgs and count_messages_tokens(system_msgs + protected_msgs) > max_tokens:
        protected_msgs.pop(0)

    return system_msgs + protected_msgs

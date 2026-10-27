"""大模型 API 封装服务 - 数据模型"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    """聊天消息"""

    role: Literal["system", "user", "assistant"] = "user"
    content: str


class TokenUsage(BaseModel):
    """Token 使用量"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatResponse(BaseModel):
    """聊天响应"""

    content: str
    model: str
    usage: TokenUsage = Field(default_factory=TokenUsage)
    request_id: str | None = None
    latency_ms: float = 0.0
    retries: int = 0


class LLMConfig(BaseModel):
    """LLM 客户端配置"""

    api_key: str
    base_url: str
    model: str
    timeout: float = 30.0
    connect_timeout: float = 10.0
    max_retries: int = 3
    retry_delay: float = 1.0
    max_retry_delay: float = 30.0
    max_context_tokens: int = 8192
    auto_trim: bool = False  # 超限时是否自动裁剪上下文

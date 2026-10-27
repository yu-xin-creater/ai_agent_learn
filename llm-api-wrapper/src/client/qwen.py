"""Qwen（通义千问）客户端

使用 dashscope compatible-mode 端点，兼容 OpenAI Chat Completions 协议。
"""

from __future__ import annotations

from src.client.base import LLMClient
from src.models import LLMConfig


# Qwen 默认配置
QWEN_DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_DEFAULT_MODEL = "qwen-turbo"


class QwenClient(LLMClient):
    """通义千问 API 客户端

    Usage:
        config = LLMConfig(
            api_key="your-api-key",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model="qwen-turbo",
        )
        async with QwenClient(config) as client:
            response = await client.chat([
                {"role": "user", "content": "你好"}
            ])
            print(response.content)
    """

    def __init__(self, config: LLMConfig | None = None, *, api_key: str | None = None) -> None:
        """初始化 Qwen 客户端

        Args:
            config: 完整配置对象
            api_key: 快捷方式，仅提供 api_key 时使用默认配置
        """
        if config is None:
            if api_key is None:
                raise ValueError("必须提供 config 或 api_key")
            config = LLMConfig(
                api_key=api_key,
                base_url=QWEN_DEFAULT_BASE_URL,
                model=QWEN_DEFAULT_MODEL,
            )
        super().__init__(config)

    def _get_default_model(self) -> str:
        return QWEN_DEFAULT_MODEL

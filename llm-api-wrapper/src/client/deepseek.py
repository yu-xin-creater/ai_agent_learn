"""DeepSeek 客户端

完全兼容 OpenAI Chat Completions 协议。
"""

from __future__ import annotations

from src.client.base import LLMClient
from src.models import LLMConfig


# DeepSeek 默认配置
DEEPSEEK_DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_DEFAULT_MODEL = "deepseek-chat"


class DeepSeekClient(LLMClient):
    """DeepSeek API 客户端

    Usage:
        config = LLMConfig(
            api_key="your-api-key",
            base_url="https://api.deepseek.com/v1",
            model="deepseek-chat",
        )
        async with DeepSeekClient(config) as client:
            response = await client.chat([
                {"role": "user", "content": "你好"}
            ])
            print(response.content)
    """

    def __init__(self, config: LLMConfig | None = None, *, api_key: str | None = None) -> None:
        """初始化 DeepSeek 客户端

        Args:
            config: 完整配置对象
            api_key: 快捷方式，仅提供 api_key 时使用默认配置
        """
        if config is None:
            if api_key is None:
                raise ValueError("必须提供 config 或 api_key")
            config = LLMConfig(
                api_key=api_key,
                base_url=DEEPSEEK_DEFAULT_BASE_URL,
                model=DEEPSEEK_DEFAULT_MODEL,
            )
        super().__init__(config)

    def _get_default_model(self) -> str:
        return DEEPSEEK_DEFAULT_MODEL

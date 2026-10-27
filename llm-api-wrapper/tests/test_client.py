"""Client 模块测试

使用 respx mock HTTP 请求，验证客户端行为。
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from src.client import DeepSeekClient, QwenClient
from src.exceptions import (
    LLMAuthError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
    LLMTokenLimitError,
)
from src.models import LLMConfig


@pytest.fixture
def qwen_config() -> LLMConfig:
    return LLMConfig(
        api_key="test-key",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen-turbo",
        max_retries=1,
        retry_delay=0.01,
        max_context_tokens=8192,
    )


@pytest.fixture
def deepseek_config() -> LLMConfig:
    return LLMConfig(
        api_key="test-key",
        base_url="https://api.deepseek.com/v1",
        model="deepseek-chat",
        max_retries=1,
        retry_delay=0.01,
        max_context_tokens=8192,
    )


def _make_success_response(content: str = "你好！", model: str = "qwen-turbo") -> dict:
    return {
        "id": "chatcmpl-test",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }


class TestQwenClient:
    """QwenClient 单元测试"""

    @respx.mock
    @pytest.mark.asyncio
    async def test_successful_chat(self, qwen_config):
        respx.post(
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        ).mock(return_value=httpx.Response(200, json=_make_success_response()))

        async with QwenClient(qwen_config) as client:
            response = await client.chat([{"role": "user", "content": "你好"}])

        assert response.content == "你好！"
        assert response.model == "qwen-turbo"
        assert response.usage.total_tokens == 15
        assert response.latency_ms > 0

    @respx.mock
    @pytest.mark.asyncio
    async def test_timeout_error(self, qwen_config):
        qwen_config.timeout = 0.1
        qwen_config.max_retries = 1

        respx.post(
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        ).mock(side_effect=httpx.ReadTimeout("timeout"))

        async with QwenClient(qwen_config) as client:
            with pytest.raises(LLMTimeoutError):
                await client.chat([{"role": "user", "content": "test"}])

    @respx.mock
    @pytest.mark.asyncio
    async def test_rate_limit_error(self, qwen_config):
        qwen_config.max_retries = 1

        respx.post(
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        ).mock(
            return_value=httpx.Response(
                429,
                json={"error": {"message": "Rate limit exceeded"}},
                headers={"retry-after": "1"},
            )
        )

        async with QwenClient(qwen_config) as client:
            with pytest.raises(LLMRateLimitError) as exc_info:
                await client.chat([{"role": "user", "content": "test"}])
            assert exc_info.value.retry_after == 1.0

    @respx.mock
    @pytest.mark.asyncio
    async def test_auth_error(self, qwen_config):
        respx.post(
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        ).mock(
            return_value=httpx.Response(
                401, json={"error": {"message": "Invalid API key"}}
            )
        )

        async with QwenClient(qwen_config) as client:
            with pytest.raises(LLMAuthError):
                await client.chat([{"role": "user", "content": "test"}])

    @respx.mock
    @pytest.mark.asyncio
    async def test_server_error(self, qwen_config):
        qwen_config.max_retries = 1

        respx.post(
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        ).mock(
            return_value=httpx.Response(
                500, json={"error": {"message": "Internal error"}}
            )
        )

        async with QwenClient(qwen_config) as client:
            with pytest.raises(LLMResponseError) as exc_info:
                await client.chat([{"role": "user", "content": "test"}])
            assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_token_limit_error(self, qwen_config):
        qwen_config.max_context_tokens = 10  # 极小值触发超限

        async with QwenClient(qwen_config) as client:
            with pytest.raises(LLMTokenLimitError) as exc_info:
                await client.chat([{"role": "user", "content": "这是一段很长的文本" * 10}])
            assert exc_info.value.token_count is not None
            assert exc_info.value.max_tokens == 10

    @respx.mock
    @pytest.mark.asyncio
    async def test_auto_trim(self, qwen_config):
        qwen_config.max_context_tokens = 50
        qwen_config.auto_trim = True

        respx.post(
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        ).mock(return_value=httpx.Response(200, json=_make_success_response()))

        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "old " * 100},
            {"role": "user", "content": "new"},
        ]

        async with QwenClient(qwen_config) as client:
            # 应不抛异常，而是自动裁剪
            response = await client.chat(messages)
            assert response.content == "你好！"

    @respx.mock
    @pytest.mark.asyncio
    async def test_request_hook(self, qwen_config):
        respx.post(
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        ).mock(return_value=httpx.Response(200, json=_make_success_response()))

        hook_calls = []

        async with QwenClient(qwen_config) as client:
            client.on_request(lambda body: hook_calls.append(body))
            await client.chat([{"role": "user", "content": "test"}])

        assert len(hook_calls) == 1
        assert hook_calls[0]["model"] == "qwen-turbo"

    @respx.mock
    @pytest.mark.asyncio
    async def test_response_hook(self, qwen_config):
        respx.post(
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        ).mock(return_value=httpx.Response(200, json=_make_success_response()))

        hook_calls = []

        async with QwenClient(qwen_config) as client:
            client.on_response(lambda resp: hook_calls.append(resp))
            await client.chat([{"role": "user", "content": "test"}])

        assert len(hook_calls) == 1
        assert hook_calls[0].content == "你好！"


class TestDeepSeekClient:
    """DeepSeekClient 单元测试"""

    @respx.mock
    @pytest.mark.asyncio
    async def test_successful_chat(self, deepseek_config):
        respx.post("https://api.deepseek.com/v1/chat/completions").mock(
            return_value=httpx.Response(
                200, json=_make_success_response("Hello!", "deepseek-chat")
            )
        )

        async with DeepSeekClient(deepseek_config) as client:
            response = await client.chat([{"role": "user", "content": "Hi"}])

        assert response.content == "Hello!"
        assert response.model == "deepseek-chat"

    @respx.mock
    @pytest.mark.asyncio
    async def test_structured_output(self, deepseek_config):
        from pydantic import BaseModel

        class UserInfo(BaseModel):
            name: str
            age: int

        structured_response = _make_success_response(
            '{"name": "Alice", "age": 30}', "deepseek-chat"
        )

        respx.post("https://api.deepseek.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=structured_response)
        )

        async with DeepSeekClient(deepseek_config) as client:
            result = await client.chat_structured(
                [{"role": "user", "content": "用户信息"}],
                response_model=UserInfo,
            )

        assert isinstance(result, UserInfo)
        assert result.name == "Alice"
        assert result.age == 30

    def test_init_with_api_key(self):
        client = DeepSeekClient(api_key="test-key")
        assert client.config.api_key == "test-key"
        assert "deepseek" in client.config.base_url

    def test_init_without_args_raises(self):
        with pytest.raises(ValueError):
            DeepSeekClient()


class TestQwenClientInit:
    """客户端初始化测试"""

    def test_init_with_api_key(self):
        client = QwenClient(api_key="test-key")
        assert client.config.api_key == "test-key"
        assert "dashscope" in client.config.base_url

    def test_init_without_args_raises(self):
        with pytest.raises(ValueError):
            QwenClient()

    def test_init_with_config(self, qwen_config):
        client = QwenClient(qwen_config)
        assert client.config == qwen_config

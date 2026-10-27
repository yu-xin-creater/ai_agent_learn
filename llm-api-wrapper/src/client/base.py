"""LLM 客户端抽象基类

定义统一的调用接口，子类只需实现 _build_request_body 和配置差异。
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Callable, TypeVar

import httpx
from pydantic import BaseModel

from src.exceptions import (
    LLMAuthError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
    LLMTokenLimitError,
)
from src.models import ChatResponse, LLMConfig, Message, TokenUsage
from src.utils.context_trimmer import trim_context
from src.utils.retry import llm_retry
from src.utils.structured_output import (
    build_validation_feedback,
    inject_schema_into_messages,
    parse_structured_response,
)
from src.utils.tokens import count_messages_tokens

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# 钩子回调类型
RequestHook = Callable[[dict[str, Any]], None]
ResponseHook = Callable[[ChatResponse], None]


class LLMClient(ABC):
    """LLM 客户端抽象基类

    提供统一的 chat / chat_structured / chat_sync 接口。
    子类需实现 _get_default_model() 和可选覆盖 _build_headers()。
    """

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._client: httpx.AsyncClient | None = None
        self._on_request: RequestHook | None = None
        self._on_response: ResponseHook | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        """懒初始化 httpx 异步客户端"""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                headers=self._build_headers(),
                timeout=httpx.Timeout(
                    connect=self.config.connect_timeout,
                    read=self.config.timeout,
                    write=self.config.timeout,
                    pool=self.config.timeout,
                ),
            )
        return self._client

    def on_request(self, hook: RequestHook) -> None:
        """注册请求钩子"""
        self._on_request = hook

    def on_response(self, hook: ResponseHook) -> None:
        """注册响应钩子"""
        self._on_response = hook

    def _build_headers(self) -> dict[str, str]:
        """构建请求头，子类可覆盖"""
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

    @abstractmethod
    def _get_default_model(self) -> str:
        """返回默认模型名称"""
        ...

    def _build_request_body(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stream: bool = False,
        response_format: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """构建请求体（OpenAI 兼容格式）"""
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if response_format is not None:
            body["response_format"] = response_format
        return body

    async def chat(
        self,
        messages: list[Message | dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        timeout: float | None = None,
        stream: bool = False,
    ) -> ChatResponse:
        """发送聊天请求

        Args:
            messages: 消息列表
            model: 模型名称，None 使用配置默认值
            temperature: 温度参数
            max_tokens: 最大生成 token 数
            timeout: 本次请求超时（覆盖配置）
            stream: 是否流式（当前仅支持非流式）

        Returns:
            ChatResponse 包含内容、usage、延迟等信息

        Raises:
            LLMTokenLimitError: prompt token 超限且未开启 auto_trim
            LLMTimeoutError: 请求超时
            LLMRateLimitError: 429 限流
            LLMAuthError: 401/403 认证失败
            LLMResponseError: 其他 HTTP 错误或响应解析失败
        """
        use_model = model or self.config.model or self._get_default_model()

        # 标准化消息格式
        msg_dicts = self._normalize_messages(messages)

        # Token 预估与裁剪
        msg_dicts = self._check_and_trim_tokens(msg_dicts, use_model)

        # 构建请求体
        body = self._build_request_body(
            msg_dicts,
            model=use_model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
        )

        # 触发请求钩子
        if self._on_request:
            self._on_request(body)

        # 执行请求（带重试）
        response = await self._do_request(body, timeout=timeout, model=use_model)

        # 触发响应钩子
        if self._on_response:
            self._on_response(response)

        return response

    async def chat_structured(
        self,
        messages: list[Message | dict[str, str]],
        response_model: type[T],
        *,
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int | None = None,
        max_retries: int = 2,
        use_native_json_mode: bool = True,
        **kwargs: Any,
    ) -> T:
        """结构化输出：强制模型返回符合 Pydantic schema 的 JSON

        Args:
            messages: 消息列表
            response_model: 期望的 Pydantic 模型类
            model: 模型名称
            temperature: 温度（结构化输出建议较低）
            max_tokens: 最大生成 token 数
            max_retries: 校验失败后的最大重试次数
            use_native_json_mode: 是否使用模型原生 response_format

        Returns:
            校验通过的 Pydantic 模型实例

        Raises:
            LLMResponseError: 多次重试后仍无法获得合法结构化输出
        """
        use_model = model or self.config.model or self._get_default_model()
        msg_dicts = self._normalize_messages(messages)

        # 注入 schema 到 prompt
        msg_dicts = inject_schema_into_messages(msg_dicts, response_model)

        last_error: LLMResponseError | None = None

        for attempt in range(1 + max_retries):
            # 构建请求
            response_format = {"type": "json_object"} if use_native_json_mode else None
            body = self._build_request_body(
                msg_dicts,
                model=use_model,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            )

            response = await self._do_request(body, model=use_model)

            try:
                result = parse_structured_response(response.content, response_model)
                return result
            except LLMResponseError as e:
                last_error = e
                logger.warning(
                    "结构化输出校验失败 | attempt=%d/%d | error=%s",
                    attempt + 1,
                    1 + max_retries,
                    e.message,
                )
                if attempt < max_retries:
                    # 追加错误反馈到消息中重试
                    feedback = build_validation_feedback(e)
                    msg_dicts = msg_dicts + [
                        {"role": "assistant", "content": response.content},
                        {"role": "user", "content": feedback},
                    ]

        # 所有重试耗尽
        raise LLMResponseError(
            f"结构化输出在 {1 + max_retries} 次尝试后仍失败",
            validation_errors=last_error.validation_errors if last_error else None,
            model=use_model,
            raw_response=last_error.raw_response if last_error else None,
        )

    def chat_sync(
        self,
        messages: list[Message | dict[str, str]],
        **kwargs: Any,
    ) -> ChatResponse:
        """同步版本的 chat，通过 asyncio.run 桥接"""
        return asyncio.run(self.chat(messages, **kwargs))

    async def close(self) -> None:
        """关闭 HTTP 客户端"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def __aenter__(self) -> "LLMClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    # ──────────────────────────── 内部方法 ────────────────────────────

    def _normalize_messages(
        self, messages: list[Message | dict[str, str]]
    ) -> list[dict[str, str]]:
        """将 Message 对象统一转为 dict"""
        result = []
        for msg in messages:
            if isinstance(msg, Message):
                result.append({"role": msg.role, "content": msg.content})
            else:
                result.append(msg)
        return result

    def _check_and_trim_tokens(
        self, messages: list[dict[str, str]], model: str
    ) -> list[dict[str, str]]:
        """检查 token 限制，按需裁剪或报错"""
        token_count = count_messages_tokens(messages)
        max_tokens = self.config.max_context_tokens

        if token_count <= max_tokens:
            return messages

        if self.config.auto_trim:
            logger.info(
                "Token 超限，执行自动裁剪 | tokens=%d | max=%d",
                token_count,
                max_tokens,
            )
            return trim_context(messages, max_tokens)

        raise LLMTokenLimitError(
            token_count=token_count,
            max_tokens=max_tokens,
            model=model,
        )

    @llm_retry(max_attempts=3)
    async def _do_request(
        self,
        body: dict[str, Any],
        *,
        timeout: float | None = None,
        model: str | None = None,
    ) -> ChatResponse:
        """执行 HTTP 请求并解析响应（带重试装饰器）"""
        start_time = time.perf_counter()
        use_timeout = timeout or self.config.timeout

        try:
            resp = await self.client.post(
                "/chat/completions",
                json=body,
                timeout=httpx.Timeout(
                    connect=self.config.connect_timeout,
                    read=use_timeout,
                    write=use_timeout,
                    pool=use_timeout,
                ),
            )
        except httpx.TimeoutException as e:
            raise LLMTimeoutError(timeout=use_timeout, model=model) from e
        except httpx.ConnectError as e:
            raise LLMResponseError(
                f"连接失败: {e}", status_code=None, model=model
            ) from e

        latency_ms = (time.perf_counter() - start_time) * 1000

        # 处理 HTTP 错误
        if resp.status_code != 200:
            self._raise_for_status(resp, model=model)

        # 解析响应
        try:
            data = resp.json()
        except Exception as e:
            raise LLMResponseError(
                f"响应 JSON 解析失败: {e}",
                status_code=resp.status_code,
                model=model,
                raw_response=resp.text,
            ) from e

        return self._parse_response(data, latency_ms=latency_ms, model=model)

    def _raise_for_status(
        self, resp: httpx.Response, *, model: str | None = None
    ) -> None:
        """根据 HTTP 状态码抛出对应异常"""
        status = resp.status_code
        request_id = resp.headers.get("x-request-id")

        try:
            body = resp.json()
            error_msg = body.get("error", {}).get("message", resp.text)
        except Exception:
            error_msg = resp.text

        kwargs: dict[str, Any] = {
            "status_code": status,
            "request_id": request_id,
            "model": model,
            "raw_response": resp.text,
        }

        if status == 429:
            retry_after = resp.headers.get("retry-after")
            raise LLMRateLimitError(
                message=f"速率限制: {error_msg}",
                retry_after=float(retry_after) if retry_after else None,
                request_id=request_id,
                model=model,
                raw_response=resp.text,
            )
        elif status in (401, 403):
            raise LLMAuthError(message=f"认证失败: {error_msg}", **kwargs)
        else:
            raise LLMResponseError(message=f"HTTP {status}: {error_msg}", **kwargs)

    def _parse_response(
        self,
        data: dict[str, Any],
        *,
        latency_ms: float,
        model: str | None = None,
    ) -> ChatResponse:
        """从 API 原始响应解析出 ChatResponse"""
        try:
            choices = data.get("choices", [])
            if not choices:
                raise LLMResponseError(
                    "响应中无 choices",
                    model=model,
                    raw_response=data,
                )

            content = choices[0].get("message", {}).get("content", "")
            usage_data = data.get("usage", {})
            request_id = data.get("id")

            usage = TokenUsage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            )

            response = ChatResponse(
                content=content,
                model=data.get("model", model or ""),
                usage=usage,
                request_id=request_id,
                latency_ms=latency_ms,
            )

            logger.info(
                "LLM 请求完成 | model=%s | tokens=%d | latency=%.0fms",
                response.model,
                usage.total_tokens,
                latency_ms,
            )

            return response

        except LLMResponseError:
            raise
        except Exception as e:
            raise LLMResponseError(
                f"响应解析失败: {e}",
                model=model,
                raw_response=data,
            ) from e

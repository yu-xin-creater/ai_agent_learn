"""大模型 API 封装服务 - 异常体系

层次化异常树，所有异常携带请求上下文信息便于排查。
"""

from __future__ import annotations

from typing import Any


class LLMError(Exception):
    """所有 LLM 相关错误的基类"""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        request_id: str | None = None,
        model: str | None = None,
        raw_response: Any = None,
    ) -> None:
        self.message = message
        self.status_code = status_code
        self.request_id = request_id
        self.model = model
        self.raw_response = raw_response
        super().__init__(message)

    def __repr__(self) -> str:
        attrs = []
        if self.status_code is not None:
            attrs.append(f"status_code={self.status_code}")
        if self.request_id:
            attrs.append(f"request_id={self.request_id!r}")
        if self.model:
            attrs.append(f"model={self.model!r}")
        extra = f" ({', '.join(attrs)})" if attrs else ""
        return f"{self.__class__.__name__}: {self.message}{extra}"

    def __str__(self) -> str:
        return self.__repr__()


class LLMTimeoutError(LLMError):
    """请求超时

    当 HTTP 请求超过配置的 timeout 时间时触发。
    """

    def __init__(
        self,
        message: str = "请求超时",
        *,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> None:
        self.timeout = timeout
        if timeout is not None:
            message = f"{message} (timeout={timeout}s)"
        super().__init__(message, **kwargs)


class LLMRateLimitError(LLMError):
    """触发速率限制 (HTTP 429)

    当 API 返回 429 Too Many Requests 时触发。
    包含 retry_after 信息（如果响应头中有提供）。
    """

    def __init__(
        self,
        message: str = "触发速率限制",
        *,
        retry_after: float | None = None,
        **kwargs: Any,
    ) -> None:
        self.retry_after = retry_after
        if retry_after is not None:
            message = f"{message} (retry_after={retry_after}s)"
        super().__init__(message, status_code=429, **kwargs)


class LLMAuthError(LLMError):
    """认证/授权失败 (HTTP 401 / 403)

    API Key 无效、过期或权限不足时触发。此类错误不应重试。
    """

    def __init__(
        self,
        message: str = "认证失败，请检查 API Key",
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)


class LLMResponseError(LLMError):
    """非预期响应格式或 HTTP 4xx/5xx

    包括：响应体无法解析、缺少必要字段、结构化输出校验失败等。
    """

    def __init__(
        self,
        message: str = "响应格式异常",
        *,
        validation_errors: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        self.validation_errors = validation_errors
        super().__init__(message, **kwargs)


class LLMTokenLimitError(LLMError):
    """Token 超限

    当 prompt tokens 超过模型的上下文窗口大小，
    且自动裁剪未启用或裁剪后仍超限时触发。
    """

    def __init__(
        self,
        message: str = "Token 数量超过模型上下文窗口限制",
        *,
        token_count: int | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.token_count = token_count
        self.max_tokens = max_tokens
        if token_count is not None and max_tokens is not None:
            message = f"{message} ({token_count}/{max_tokens})"
        super().__init__(message, **kwargs)

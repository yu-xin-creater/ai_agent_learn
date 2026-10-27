"""重试工具

基于 tenacity 提供指数退避重试装饰器，统一可重试异常判定逻辑。
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
    before_sleep_log,
    RetryCallState,
)

from src.exceptions import (
    LLMError,
    LLMTimeoutError,
    LLMRateLimitError,
    LLMResponseError,
    LLMAuthError,
)

logger = logging.getLogger(__name__)

# 可重试的 HTTP 状态码
RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})


def is_retryable_error(error: BaseException) -> bool:
    """判断异常是否可重试

    可重试条件：
    - LLMTimeoutError（网络超时）
    - LLMRateLimitError（429 限流）
    - LLMResponseError 且 status_code 属于 5xx
    - 不重试 LLMAuthError（认证失败是确定性错误）
    """
    if isinstance(error, LLMAuthError):
        return False
    if isinstance(error, (LLMTimeoutError, LLMRateLimitError)):
        return True
    if isinstance(error, LLMResponseError):
        return error.status_code in RETRYABLE_STATUS_CODES if error.status_code else False
    if isinstance(error, LLMError):
        return error.status_code in RETRYABLE_STATUS_CODES if error.status_code else False
    return False


def _log_retry(retry_state: RetryCallState) -> None:
    """重试前记录日志"""
    attempt = retry_state.attempt_number
    wait = retry_state.next_action.sleep if retry_state.next_action else 0  # type: ignore[union-attr]
    outcome = retry_state.outcome
    error = outcome.exception() if outcome else None
    logger.warning(
        "LLM 请求重试 | attempt=%d | wait=%.1fs | error=%s",
        attempt,
        wait,
        repr(error),
    )


def llm_retry(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 30.0,
    jitter: float = 1.0,
) -> Callable[[Any], Any]:
    """创建 LLM 请求重试装饰器

    Args:
        max_attempts: 最大尝试次数（含首次）
        initial_delay: 首次重试等待秒数
        max_delay: 最大等待秒数
        jitter: 随机抖动范围

    Returns:
        tenacity retry 装饰器

    Usage:
        @llm_retry(max_attempts=3)
        async def make_request(...):
            ...
    """
    return retry(
        retry=retry_if_exception(is_retryable_error),
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential_jitter(
            initial=initial_delay,
            max=max_delay,
            jitter=jitter,
        ),
        before_sleep=_log_retry,
        reraise=True,
    )

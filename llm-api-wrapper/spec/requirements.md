# 需求文档：大模型 API 封装服务

## 1. 项目概述

构建一个 Python SDK，统一封装 Qwen（通义千问）和 DeepSeek 两类大模型 API，为上层业务提供稳定、可观测、可控的调用层。

## 2. 功能需求

### FR-1: 多模型统一接口

- 提供统一的 `LLMClient` 抽象基类，子类 `QwenClient` / `DeepSeekClient` 各自实现具体 HTTP 调用。
- 支持同步与异步（async/await）两种调用方式。
- 支持流式（streaming）与非流式响应。

### FR-2: 超时控制

- 每次请求支持自定义 `timeout` 参数（秒），默认 30s。
- 连接超时与读取超时分离配置。
- 超时触发 `LLMTimeoutError` 异常，包含请求上下文信息。

### FR-3: 自动重试

- 采用指数退避策略（exponential backoff），默认最多 3 次重试。
- 仅对可重试的错误码（429 / 500 / 502 / 503 / 504）进行重试。
- 重试次数、初始延迟、最大延迟均可配置。
- 每次重试记录日志，包含 attempt 序号与等待时间。

### FR-4: Token 计数

- 请求前预估 prompt tokens 数量（基于 tiktoken）。
- 响应后解析实际 usage（prompt_tokens + completion_tokens + total_tokens）。
- 当 prompt tokens 超过模型上下文窗口时，触发自动裁剪或抛出 `LLMTokenLimitError`。

### FR-5: 上下文裁剪（Context Trimming）

- 提供 `trim_context(messages, max_tokens, strategy)` 工具函数。
- 裁剪策略：
  - `oldest_first`：从最早的消息开始移除（保留 system prompt 和最近 N 轮）。
  - `summarize`：对被裁剪部分生成摘要后插回（可选，依赖二次 LLM 调用）。
- 裁剪后保证 token 数不超过 `max_tokens`。

### FR-6: 结构化 JSON 输出

- 通过 Pydantic BaseModel 定义期望输出 schema。
- 在 prompt 中注入 JSON schema 描述，强制模型输出 JSON。
- 对响应进行 Pydantic 校验，校验失败则自动重试（最多 2 次）。
- 支持 `response_format={"type": "json_object"}` 模式（模型原生支持时优先使用）。

### FR-7: 异常体系

| 异常类 | 触发条件 |
|--------|---------|
| `LLMError` | 所有 LLM 相关错误的基类 |
| `LLMTimeoutError` | 请求超时 |
| `LLMRateLimitError` | HTTP 429 |
| `LLMAuthError` | HTTP 401 / 403 |
| `LLMResponseError` | 非预期响应格式或 HTTP 4xx/5xx |
| `LLMTokenLimitError` | Token 超限 |

### FR-8: 可观测性

- 每次请求记录结构化日志：model、tokens、latency_ms、retries、status。
- 暴露 `on_request` / `on_response` 钩子回调，便于接入监控。

## 3. 非功能需求

| 编号 | 描述 |
|------|------|
| NFR-1 | Python ≥ 3.10 |
| NFR-2 | 全异步 IO（httpx），同步接口通过 `asyncio.run` 桥接 |
| NFR-3 | 零硬编码密钥，通过环境变量 / .env 注入 |
| NFR-4 | 单元测试覆盖率 ≥ 90%（pytest + respx mock） |
| NFR-5 | PBT 测试覆盖 token 计数、上下文裁剪、结构化输出校验 |
| NFR-6 | 代码风格 ruff 检查通过 |

## 4. 约束与假设

- Qwen API 兼容 OpenAI Chat Completions 协议（dashscope compatible-mode）。
- DeepSeek API 完全兼容 OpenAI Chat Completions 协议。
- tiktoken 使用 `cl100k_base` 编码作为 token 预估基准（与实际模型 tokenizer 存在误差，误差 ≤ 5%）。

## 5. 验收标准

1. `QwenClient` 和 `DeepSeekClient` 均可正确完成非流式 chat completion 调用。
2. 超时 → 抛出 `LLMTimeoutError`，重试耗尽 → 抛出最后一次异常。
3. 结构化输出模式下，返回值为合法 Pydantic 模型实例。
4. `trim_context` 裁剪后 token 数 ≤ 指定阈值。
5. PBT 测试能检测到模型输出与 schema 不匹配的情况（模拟幻觉）。
6. 所有测试在无网络环境下通过（全量 mock）。

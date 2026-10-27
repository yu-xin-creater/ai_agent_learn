# 架构设计文档：大模型 API 封装服务

## 1. 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                    Application Layer                      │
│              (业务代码调用 LLMClient)                      │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                   Client Layer (src/client/)              │
│  ┌─────────────┐  ┌─────────────┐  ┌────────────────┐  │
│  │ LLMClient   │  │ QwenClient  │  │ DeepSeekClient │  │
│  │ (ABC)       │◄─┤ (具体实现)   │  │ (具体实现)      │  │
│  └─────────────┘  └─────────────┘  └────────────────┘  │
└────────────┬───────────────┬────────────────────────────┘
             │               │
┌────────────▼───────────────▼────────────────────────────┐
│                   Utils Layer (src/utils/)                │
│  ┌──────────┐ ┌──────────┐ ┌─────────┐ ┌────────────┐  │
│  │ token    │ │ context  │ │ retry   │ │ structured │  │
│  │ counter  │ │ trimmer  │ │ handler │ │ output     │  │
│  └──────────┘ └──────────┘ └─────────┘ └────────────┘  │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│              Exceptions (src/exceptions.py)               │
│  LLMError ─┬─ LLMTimeoutError                           │
│            ├─ LLMRateLimitError                          │
│            ├─ LLMAuthError                               │
│            ├─ LLMResponseError                           │
│            └─ LLMTokenLimitError                         │
└─────────────────────────────────────────────────────────┘
```

## 2. 模块职责

### 2.1 Client Layer (`src/client/`)

| 文件 | 职责 |
|------|------|
| `base.py` | 定义 `LLMClient` 抽象基类：统一接口签名、公共配置、钩子回调 |
| `qwen.py` | Qwen（通义千问）具体实现，dashscope compatible-mode 端点 |
| `deepseek.py` | DeepSeek 具体实现，标准 OpenAI 兼容端点 |
| `__init__.py` | 导出公共符号 |

**核心接口设计：**

```python
class LLMClient(ABC):
    async def chat(
        self,
        messages: list[Message],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        timeout: float | None = None,
        response_model: type[BaseModel] | None = None,  # 结构化输出
        stream: bool = False,
    ) -> ChatResponse | AsyncIterator[ChatChunk]:
        ...

    async def chat_structured(
        self,
        messages: list[Message],
        response_model: type[BaseModel],
        **kwargs,
    ) -> BaseModel:
        """结构化输出便捷方法，内含 schema 注入 + 校验重试"""
        ...
```

### 2.2 Utils Layer (`src/utils/`)

| 文件 | 职责 |
|------|------|
| `tokens.py` | Token 计数：基于 tiktoken 的 `count_tokens(text)` / `count_messages_tokens(messages)` |
| `context_trimmer.py` | 上下文裁剪：`trim_context(messages, max_tokens, strategy)` |
| `retry.py` | 重试装饰器：封装 tenacity，统一指数退避 + 可重试判定 |
| `structured_output.py` | 结构化输出：schema 注入 prompt、响应解析与校验 |

### 2.3 Exceptions (`src/exceptions.py`)

层次化异常树，所有异常携带：
- `status_code: int | None` — HTTP 状态码
- `request_id: str | None` — 请求追踪 ID
- `model: str | None` — 使用的模型名
- `raw_response: Any` — 原始响应体

### 2.4 数据模型 (`src/models.py`)

```python
class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str

class TokenUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatResponse(BaseModel):
    content: str
    model: str
    usage: TokenUsage
    request_id: str | None = None
    latency_ms: float
    retries: int = 0
```

## 3. 关键流程

### 3.1 标准请求流程

```
chat() 调用
  │
  ├─ 1. Token 预估 → 超限？ → trim_context() 或 raise LLMTokenLimitError
  │
  ├─ 2. 构造 HTTP 请求（headers + body）
  │
  ├─ 3. 发送请求（带 retry 装饰器）
  │     ├─ 超时 → LLMTimeoutError（可重试）
  │     ├─ 429 → LLMRateLimitError（可重试）
  │     ├─ 401/403 → LLMAuthError（不重试）
  │     └─ 5xx → LLMResponseError（可重试）
  │
  ├─ 4. 解析响应 → ChatResponse
  │
  ├─ 5. 触发 on_response 钩子（记录日志/指标）
  │
  └─ 6. 返回结果
```

### 3.2 结构化输出流程

```
chat_structured() 调用
  │
  ├─ 1. 从 response_model 生成 JSON Schema 描述
  │
  ├─ 2. 注入 schema 到 system prompt（或使用 response_format）
  │
  ├─ 3. 调用 chat() 获取响应
  │
  ├─ 4. 提取 JSON → Pydantic parse
  │     ├─ 成功 → 返回模型实例
  │     └─ 失败 → 重试（最多 2 次，附带错误反馈）
  │
  └─ 5. 重试耗尽 → raise LLMResponseError
```

### 3.3 上下文裁剪流程

```
trim_context(messages, max_tokens, strategy="oldest_first")
  │
  ├─ 1. 计算当前总 tokens
  │
  ├─ 2. 如果 ≤ max_tokens → 原样返回
  │
  ├─ 3. 分离 system 消息 + 对话消息
  │
  ├─ 4. oldest_first 策略：
  │     └─ 从最早的对话消息开始移除，直到 tokens ≤ max_tokens
  │
  └─ 5. 返回裁剪后的 messages
```

## 4. 重试策略详细设计

```python
RETRY_CONFIG = {
    "max_attempts": 3,          # 最大重试次数
    "initial_delay": 1.0,       # 首次重试等待（秒）
    "max_delay": 30.0,          # 最大等待时间
    "exponential_base": 2,      # 指数基数
    "jitter": True,             # 添加随机抖动
    "retryable_status_codes": {429, 500, 502, 503, 504},
}
```

## 5. 配置管理

通过 `pydantic-settings` 风格管理配置：

```python
class LLMConfig(BaseModel):
    api_key: str
    base_url: str
    model: str
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    max_context_tokens: int = 8192  # 模型窗口大小
```

## 6. 依赖关系

```
httpx          → 异步 HTTP 客户端
tiktoken       → Token 计数
tenacity       → 重试机制
pydantic       → 数据校验 + 结构化输出 schema
python-dotenv  → 环境变量加载
```

## 7. 测试策略

| 层级 | 工具 | 覆盖内容 |
|------|------|---------|
| 单元测试 | pytest + respx | Client 请求/响应 mock、异常触发、重试行为 |
| 集成测试 | pytest + httpx mock | 完整流程：chat → token 计数 → 裁剪 → 结构化输出 |
| PBT 测试 | hypothesis | token 计数一致性、裁剪后 token ≤ 阈值、结构化输出 schema 校验、幻觉检测 |

### 7.1 PBT 幻觉校验设计

- 生成随机 JSON schema（属性名、类型、嵌套）。
- 模拟模型返回不符合 schema 的 JSON（缺少字段、类型错误、多余字段）。
- 验证系统能正确检测并拒绝不合规输出，触发重试或抛出异常。

## 8. 文件结构总览

```
llm-api-wrapper/
├── pyproject.toml
├── .env.example
├── .gitignore
├── spec/
│   ├── requirements.md
│   ├── design.md
│   └── tasks.md
├── src/
│   ├── __init__.py
│   ├── models.py
│   ├── exceptions.py
│   ├── client/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── qwen.py
│   │   └── deepseek.py
│   └── utils/
│       ├── __init__.py
│       ├── tokens.py
│       ├── context_trimmer.py
│       ├── retry.py
│       └── structured_output.py
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_client.py
    ├── test_tokens.py
    ├── test_context_trimmer.py
    ├── test_structured_output.py
    └── test_pbt_hallucination.py
```

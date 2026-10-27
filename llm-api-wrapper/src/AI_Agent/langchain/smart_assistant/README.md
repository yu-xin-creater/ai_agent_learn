# 小智智能助手 (LangChain 1.0 版)

> 本项目已从 0.x 风格重构为 **LangChain 1.0** 的 `create_agent` + **middleware** 范式。
> 改造时间: 2026-07-23

一个用于学习 LangChain 1.0 生产级 Agent 开发的示例：涵盖记忆、对话摘要、人工介入、PII 护栏、调用限流、流式输出、结构化输出等企业生产必备能力。

---

## 项目结构

```
smart_assistant/
├── config.py            # 模型构建 / LangSmith 开关 / 系统提示词
├── middlewares.py       # 自定义 middleware 示例(动态提示词/前后钩子)
├── agent_factory.py     # 核心：create_agent + 全套 middleware 组装(开关式)
├── main.py              # 交互式主入口(生产级配置)
├── requirements.txt     # 依赖(1.0)
├── README.md            # 本文档
├── tools/               # 工具集合
│   ├── weather.py       # 天气查询
│   ├── calculator.py    # 数学计算
│   ├── translator.py    # 翻译(已修复语法错误)
│   ├── knowledge.py     # 知识库FAQ查询(已补全)
│   ├── actions.py       # 敏感操作:发邮件/下单(用于HITL演示)
│   └── __init__.py      # 导出 SAFE_TOOLS / SENSITIVE_TOOLS / ALL_TOOLS
└── demos/               # 8个独立可运行的生产特性 demo
    ├── demo_01_basic_agent.py       # 最简 create_agent
    ├── demo_02_dynamic_prompt.py    # 动态系统提示词
    ├── demo_03_memory.py            # 持久化记忆
    ├── demo_04_summarization.py     # 对话摘要(控成本)
    ├── demo_05_human_in_the_loop.py # 人工介入审批
    ├── demo_06_guardrails_pii.py    # PII护栏 + 调用限流
    ├── demo_07_streaming.py         # 流式输出
    └── demo_08_structured_output.py # 结构化输出
```

---

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置密钥 (smart_assistant/.env)
#    DASHSCOPE_API_KEY=你的通义千问密钥

# 3. 运行主程序
python main.py

# 4. 运行某个特性 demo
python demos/demo_03_memory.py
```

---

## 0.x → 1.0 迁移对照（重点）

这是本次改造的核心。LangChain 1.0 把 `langchain` 包精简为「agent 核心」，
构建在 `langgraph` 运行时之上，旧的 Chain 抽象被移到了 `langchain-classic`。

| 能力 | 0.x 旧写法 | 1.0 新写法 |
|------|-----------|-----------|
| 构建 Agent | `initialize_agent(tools, llm, agent=AgentType.XXX)` | `create_agent(model, tools, middleware=[...])` |
| 系统提示词 | 固定字符串，动态信息要手工拼 messages | `@dynamic_prompt` 每次调模型前动态生成 |
| 横切能力 | Callbacks + 手写逻辑，分散难组合 | **Middleware** 统一组合、可插拔 |
| 记忆 | `ConversationBufferMemory` 等 Memory 类 | `checkpointer` + `thread_id`(基于 langgraph) |
| 对话摘要 | 自己写逻辑裁剪历史 | `SummarizationMiddleware` |
| 人工介入 | 几乎没有现成方案 | `HumanInTheLoopMiddleware` + `interrupt`/`Command` |
| 护栏 | 自行实现 | `PIIMiddleware` / `ModelCallLimitMiddleware` 等 |
| 结构化输出 | 输出解析器 OutputParser | `create_agent(response_format=PydanticModel)` |
| 流式 | `callbacks` 流式回调 | `agent.stream(..., stream_mode="messages")` |

> 迁移提示：代码里见到 `LLMChain`、`SequentialChain`、`initialize_agent`、
> `ConversationChain` 等，都是 0.x 遗留(现位于 `langchain-classic`)，新项目请勿使用。

---

## Middleware 是 1.0 的灵魂

Middleware 可组合地"插"进 agent 执行循环的各个节点前后，用来做
上下文工程(context engineering)——精确控制「什么进模型、什么时候进」。

```
用户输入
  │
  ▼
[before_model] ──→ 模型调用(dynamic_prompt在此生成系统提示词) ──→ [after_model]
                                                                    │
                                            ┌───────────────────────┤
                                            ▼                       ▼
                                        需要调工具? → 工具调用      不调 → 结束
                                        (wrap_tool_call 可拦截)
```

本项目用到的 middleware：

| Middleware | 类型 | 作用 |
|-----------|------|------|
| `time_aware_prompt` | 自定义 `@dynamic_prompt` | 系统提示词注入实时时间 |
| `audit_before/after_model` | 自定义 `@before/after_model` | 调用前后审计日志 |
| `SummarizationMiddleware` | 内置 | 长对话自动摘要，控制上下文与成本 |
| `PIIMiddleware` | 内置 | 邮箱等敏感信息脱敏 |
| `ModelCallLimitMiddleware` | 内置 | 限制调用次数，防死循环 |
| `HumanInTheLoopMiddleware` | 内置 | 敏感操作人工审批 |

LangChain 1.0 还内置了很多开箱即用的 middleware（可在
`langchain.agents.middleware` 中查看），如 `ModelFallbackMiddleware`(模型降级容错)、
`ToolRetryMiddleware`(工具重试)、`ModelRetryMiddleware`、`TodoListMiddleware` 等。

---

## 企业生产必备能力 → 对应 demo

| 生产能力 | 对应 demo / 代码 |
|---------|-----------------|
| 可观测性(追踪) | `config.enable_langsmith()`，设置 `LANGSMITH_API_KEY` |
| 记忆 / 多会话 | `demo_03_memory.py`(checkpointer + thread_id) |
| 成本控制 | `demo_04_summarization.py`(对话摘要) |
| 人工兜底 | `demo_05_human_in_the_loop.py`(敏感操作审批) |
| 安全护栏 | `demo_06_guardrails_pii.py`(PII + 限流) |
| 响应体验 | `demo_07_streaming.py`(流式输出) |
| 系统集成 | `demo_08_structured_output.py`(结构化输出) |
| 评估 | 参见 RAG 模块的评估系统(可迁移到 agent) |

---

## 说明

- 模型默认使用通义千问 `qwen-plus`(`config.py` 可改)。1.0 下接口统一，
  切换 OpenAI / Claude 只需在 `config.build_model()` 换个模型类。
- 各 demo 均带 `has_valid_key()` 守卫：没配有效密钥时会提示而不报错，
  方便先读代码理解结构。

"""
Agent 工厂（LangChain 1.0 create_agent + middleware 组装）
补充时间: 2026-07-23

本模块是改造的核心：用 1.0 的 create_agent 把「模型 + 工具 + 中间件 + 记忆」
一次性组装成一个生产级 agent，各项能力都可通过开关控制。

【新旧对照：构建 Agent】
  0.x:
      from langchain.agents import initialize_agent, AgentType
      agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, ...)
      # 定制能力弱，想加摘要/护栏/人工介入要各写一套回调，难组合

  1.0:
      from langchain.agents import create_agent
      agent = create_agent(model, tools, middleware=[...], checkpointer=...)
      # 所有横切能力统一用 middleware 组合，清晰可插拔
"""

from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    PIIMiddleware,
    SummarizationMiddleware,
)
from langgraph.checkpoint.memory import MemorySaver

from .config import SYSTEM_PROMPT, build_model
from .middlewares import time_aware_prompt
from .tools import ALL_TOOLS, SAFE_TOOLS


def create_assistant(
    *,
    enable_memory: bool = True,
    enable_summarization: bool = True,
    enable_pii_guard: bool = True,
    enable_call_limit: bool = True,
    enable_human_in_the_loop: bool = False,
    include_sensitive_tools: bool = False,
):
    """
    创建智能助手 agent

    每个开关对应一项「企业生产必备能力」，方便你逐个理解和组合：

    Args:
        enable_memory: 持久化记忆（多轮对话、断点续跑）
        enable_summarization: 对话摘要（控制上下文长度与成本）
        enable_pii_guard: PII 护栏（自动给邮箱等敏感信息脱敏）
        enable_call_limit: 模型调用次数限制（防死循环、控成本）
        enable_human_in_the_loop: 敏感工具执行前人工审批
        include_sensitive_tools: 是否挂载敏感工具（发邮件/下单）

    Returns:
        已编译的 agent（LangGraph CompiledStateGraph）
    """
    model = build_model()
    tools = ALL_TOOLS if include_sensitive_tools else SAFE_TOOLS

    # ---- 组装 middleware（顺序有意义：从前置到后置）----
    middleware = [
        time_aware_prompt,  # 自定义：动态系统提示词（注入时间）
    ]

    # 对话摘要：消息过多时自动摘要旧消息，避免上下文爆炸
    if enable_summarization:
        middleware.append(
            SummarizationMiddleware(
                model=model,
                trigger=("messages", 20),   # 超过 20 条消息触发摘要
                keep=("messages", 10),      # 保留最近 10 条原文
            )
        )

    # PII 护栏：对用户输入里的邮箱做脱敏（redact）
    if enable_pii_guard:
        middleware.append(
            PIIMiddleware("email", strategy="redact", apply_to_input=True)
        )

    # 调用次数限制：单次运行最多 10 次模型调用，超了就优雅结束
    if enable_call_limit:
        middleware.append(
            ModelCallLimitMiddleware(run_limit=10, exit_behavior="end")
        )

    # 人工介入：敏感工具（发邮件/下单）执行前中断，等人工批准
    if enable_human_in_the_loop and include_sensitive_tools:
        middleware.append(
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "send_email": True,   # True = 执行前需人工批准
                    "place_order": True,
                },
                description_prefix="⚠️ 敏感操作待确认",
            )
        )

    # ---- 记忆（checkpointer）----
    checkpointer = MemorySaver() if enable_memory else None

    # ---- 组装 agent ----
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,   # 基础提示词（动态部分由 middleware 叠加）
        middleware=middleware,
        checkpointer=checkpointer,
    )
    return agent

"""
小智 - 全能智能助手 v2.0（LangChain 1.0 重构版）
改造时间: 2026-07-23

本文件由旧版（0.x 风格）重构为 1.0 的 create_agent + middleware 范式。
核心能力（记忆/摘要/护栏/限流）全部通过 agent_factory 的 middleware 组装。

════════════════════════════════════════════════════════════════════
【0.x → 1.0 主要变化对照】

1) 构建 Agent
   0.x: initialize_agent(tools, llm, agent=AgentType.XXX)
   1.0: create_agent(model, tools, middleware=[...], checkpointer=...)

2) 横切能力（提示词/摘要/护栏/人工介入）
   0.x: 分散在 Callbacks、手写逻辑里，难组合
   1.0: 统一用 Middleware 组合，可插拔

3) 系统提示词
   0.x: 固定字符串
   1.0: @dynamic_prompt 动态生成（可注入实时信息）

4) 记忆
   0.x: ConversationBufferMemory 等 Memory 类
   1.0: LangGraph Checkpointer + thread_id（天然支持多会话、持久化）

运行:
    python main.py
    或 python -m src.AI_Agent.langchain.smart_assistant.main
════════════════════════════════════════════════════════════════════
"""

import os
import sys

# 兼容「脚本直接运行」与「模块运行」两种方式
if __package__ in (None, ""):
    _CUR = os.path.dirname(os.path.abspath(__file__))
    _PARENT = os.path.dirname(_CUR)
    if _PARENT not in sys.path:
        sys.path.insert(0, _PARENT)
    import smart_assistant  # noqa: F401
    __package__ = "smart_assistant"

from .agent_factory import create_assistant
from .config import enable_langsmith, has_valid_key


def print_banner():
    print("=" * 54)
    print("        小智 - 全能智能助手 v2.0 (LangChain 1.0)")
    print("=" * 54)
    print("能力: 天气 / 计算 / 翻译 / 知识查询")
    print("特性: 记忆 · 对话摘要 · PII护栏 · 调用限流")
    print("输入 'quit' 退出，'/new' 开启新会话\n")


def main():
    print_banner()

    # 检查 API Key
    if not has_valid_key():
        print("[!] 未配置有效的 DASHSCOPE_API_KEY")
        print("    请在 smart_assistant/.env 中设置 DASHSCOPE_API_KEY=你的密钥")
        sys.exit(1)

    # 可选：开启 LangSmith 追踪（设置了 LANGSMITH_API_KEY 才生效）
    enable_langsmith(project="smart-assistant")

    # 创建 agent（生产级特性默认开启；敏感工具+人工介入见 demos/）
    agent = create_assistant(
        enable_memory=True,
        enable_summarization=True,
        enable_pii_guard=True,
        enable_call_limit=True,
    )

    # thread_id 标识一次会话，记忆按 thread_id 隔离
    thread_id = "main"

    while True:
        try:
            user_input = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "退出"):
            print("再见！")
            break
        if user_input == "/new":
            # 换一个 thread_id 即开启全新会话（旧记忆不再带入）
            import uuid
            thread_id = f"session-{uuid.uuid4().hex[:8]}"
            print(f"[系统] 已开启新会话: {thread_id}\n")
            continue

        config = {"configurable": {"thread_id": thread_id}}
        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": user_input}]},
                config,
            )
            response = result["messages"][-1].content
            print(f"小智: {response}\n")
        except Exception as e:
            print(f"[出错] {type(e).__name__}: {e}\n")


if __name__ == "__main__":
    main()

"""
Demo 04 - 对话摘要 Middleware（控制上下文长度与成本）
补充时间: 2026-07-23

要点：长对话会让 token 不断膨胀、变贵变慢。SummarizationMiddleware 在
      消息数超过阈值时，自动把旧消息压缩成摘要，只保留最近若干条原文。

生产意义：这是控制长会话成本的关键手段之一。

运行: python demo_04_summarization.py
"""

from _bootstrap import setup_path

setup_path()

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import MemorySaver

from smart_assistant.config import build_model, has_valid_key
from smart_assistant.tools import SAFE_TOOLS


def main():
    if not has_valid_key():
        print("[!] 未配置有效 DASHSCOPE_API_KEY，无法实跑。可先阅读代码理解结构。")
        return

    model = build_model()
    agent = create_agent(
        model=model,
        tools=SAFE_TOOLS,
        system_prompt="你是智能助手。",
        middleware=[
            SummarizationMiddleware(
                model=model,
                trigger=("messages", 6),   # 演示用：超过6条消息就触发摘要
                keep=("messages", 2),      # 仅保留最近2条原文，其余压成摘要
            )
        ],
        checkpointer=MemorySaver(),
    )

    config = {"configurable": {"thread_id": "demo-summ"}}
    # 连续多轮，制造足够多的消息以触发摘要
    questions = [
        "北京天气怎么样？",
        "那上海呢？",
        "帮我算一下 128 * 56",
        "把'你好'翻译成日文",
        "深圳天气如何？",
        "刚才我都问了些什么？",   # 这轮之前应已触发摘要
    ]
    for i, q in enumerate(questions, 1):
        r = agent.invoke({"messages": [{"role": "user", "content": q}]}, config)
        print(f"[第{i}轮] 你: {q}")
        print(f"        助手: {r['messages'][-1].content[:80]}\n")

    print("(观察：随着对话变长，旧消息会被自动摘要，最后一轮仍能大致回忆起聊过的内容)")


if __name__ == "__main__":
    main()

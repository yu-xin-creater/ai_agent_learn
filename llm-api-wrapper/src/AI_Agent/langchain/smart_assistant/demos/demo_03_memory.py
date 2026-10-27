"""
Demo 03 - 持久化记忆（Checkpointer + thread_id）
补充时间: 2026-07-23

要点：1.0 的记忆由 LangGraph 的 Checkpointer 提供。
      同一个 thread_id 下的多轮对话会自动带上历史；换 thread_id 即新会话。

【新旧对照】
  0.x: ConversationBufferMemory / ConversationChain 等 Memory 类
  1.0: create_agent(checkpointer=...) + config={"configurable":{"thread_id":...}}
       天然支持多会话隔离、可换成持久化存储(SqliteSaver/PostgresSaver)

运行: python demo_03_memory.py
"""

from _bootstrap import setup_path

setup_path()

from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

from smart_assistant.config import build_model, has_valid_key
from smart_assistant.tools import SAFE_TOOLS


def main():
    if not has_valid_key():
        print("[!] 未配置有效 DASHSCOPE_API_KEY，无法实跑。可先阅读代码理解结构。")
        return

    agent = create_agent(
        model=build_model(),
        tools=SAFE_TOOLS,
        system_prompt="你是智能助手，请记住对话中的信息。",
        checkpointer=MemorySaver(),   # 开启记忆
    )

    # 同一 thread_id = 同一会话，记忆连续
    config = {"configurable": {"thread_id": "user-001"}}

    r1 = agent.invoke(
        {"messages": [{"role": "user", "content": "我叫李雷，我最喜欢的城市是杭州。"}]},
        config,
    )
    print("第1轮:", r1["messages"][-1].content)

    r2 = agent.invoke(
        {"messages": [{"role": "user", "content": "我叫什么名字？帮我查一下我最喜欢城市的天气。"}]},
        config,
    )
    print("第2轮:", r2["messages"][-1].content)
    print("\n(若第2轮能答出'李雷'并查询杭州天气，说明记忆生效)")


if __name__ == "__main__":
    main()

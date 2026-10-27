"""
Demo 01 - 最简 Agent（LangChain 1.0 create_agent）
补充时间: 2026-07-23

要点：1.0 里构建一个会用工具的 agent，只需 create_agent(model, tools)。
对比 0.x 的 initialize_agent，接口更简洁、底层基于 langgraph 运行时。

运行: python demo_01_basic_agent.py
"""

from _bootstrap import setup_path

setup_path()

from langchain.agents import create_agent

from smart_assistant.config import SYSTEM_PROMPT, build_model, has_valid_key
from smart_assistant.tools import SAFE_TOOLS


def main():
    if not has_valid_key():
        print("[!] 未配置有效 DASHSCOPE_API_KEY，无法实跑。可先阅读代码理解结构。")
        return

    # 最小组装：模型 + 工具 + 系统提示词
    agent = create_agent(
        model=build_model(),
        tools=SAFE_TOOLS,
        system_prompt=SYSTEM_PROMPT,
    )

    # 单轮调用
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "北京今天天气怎么样？"}]}
    )
    print("助手:", result["messages"][-1].content)


if __name__ == "__main__":
    main()

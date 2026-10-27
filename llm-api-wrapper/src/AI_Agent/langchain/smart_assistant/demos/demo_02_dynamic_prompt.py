"""
Demo 02 - 动态系统提示词 Middleware
补充时间: 2026-07-23

要点：用 @dynamic_prompt 让系统提示词在每次调用模型前动态生成，
      这里注入"当前时间"，使助手具备时间感知。

【新旧对照】
  0.x: 系统提示词写死；要注入动态信息得手工拼 messages
  1.0: @dynamic_prompt 装饰的函数每次调模型前执行，返回最终提示词

运行: python demo_02_dynamic_prompt.py
"""

from _bootstrap import setup_path

setup_path()

from datetime import datetime

from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt

from smart_assistant.config import build_model, has_valid_key
from smart_assistant.tools import SAFE_TOOLS


# 动态提示词：每次调模型前拼入实时时间
@dynamic_prompt
def prompt_with_time(request) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"你是智能助手。当前时间是 {now}。回答与时间相关的问题时请参考它。"


def main():
    if not has_valid_key():
        print("[!] 未配置有效 DASHSCOPE_API_KEY，无法实跑。可先阅读代码理解结构。")
        return

    agent = create_agent(
        model=build_model(),
        tools=SAFE_TOOLS,
        middleware=[prompt_with_time],   # 动态提示词通过 middleware 注入
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "现在几点了？今天是星期几？"}]}
    )
    print("助手:", result["messages"][-1].content)


if __name__ == "__main__":
    main()

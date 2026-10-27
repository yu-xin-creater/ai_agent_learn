"""
Demo 05 - 人工介入 Human-in-the-Loop（敏感操作审批）
补充时间: 2026-07-23

要点：对"有副作用"的工具（发邮件/下单/转账），在执行前中断，
      交人工审批（approve/reject/edit），避免 Agent 误操作。

机制：HumanInTheLoopMiddleware 会在触发的工具执行前调用 langgraph 的
      interrupt() 暂停图运行；拿到人工决策后用 Command(resume=...) 恢复。
      注意：使用中断必须配置 checkpointer。

决策类型：approve(批准) / reject(拒绝) / edit(改参数后执行) / respond(直接回复)

运行: python demo_05_human_in_the_loop.py
"""

from _bootstrap import setup_path

setup_path()

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from smart_assistant.config import build_model, has_valid_key
from smart_assistant.tools import ALL_TOOLS


def main():
    if not has_valid_key():
        print("[!] 未配置有效 DASHSCOPE_API_KEY，无法实跑。可先阅读代码理解结构。")
        return

    agent = create_agent(
        model=build_model(),
        tools=ALL_TOOLS,   # 含敏感工具 send_email / place_order
        system_prompt="你是办公助手，可发送邮件、提交订单。",
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "send_email": True,    # 执行前必须人工批准
                    "place_order": True,
                },
                description_prefix="⚠️ 敏感操作待确认",
            )
        ],
        checkpointer=MemorySaver(),   # 中断必须有 checkpointer
    )

    config = {"configurable": {"thread_id": "demo-hitl"}}

    # 第一步：发起会触发敏感工具的请求
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "给 boss@example.com 发封邮件，主题'周报'，内容'本周已完成RAG评估模块'。"}]},
        config,
    )

    # 检查是否被中断（等待人工审批）
    if "__interrupt__" in result:
        print("[中断] Agent 请求人工审批敏感操作：")
        for intr in result["__interrupt__"]:
            print("   ", intr.value)

        # —— 模拟人工点击"批准" —— 
        print("\n[人工] 批准执行\n")
        resumed = agent.invoke(
            Command(resume=[{"type": "approve"}]),  # 决策: 批准
            config,
        )
        print("最终结果:", resumed["messages"][-1].content)
        print("\n(若要拒绝: Command(resume=[{'type': 'reject'}]))")
    else:
        print("未触发中断，直接返回:", result["messages"][-1].content)


if __name__ == "__main__":
    main()

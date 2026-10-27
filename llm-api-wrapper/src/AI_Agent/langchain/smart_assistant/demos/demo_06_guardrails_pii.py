"""
Demo 06 - 护栏：PII 脱敏 + 调用次数限制
补充时间: 2026-07-23

要点（两类生产护栏）：
  1) PIIMiddleware：自动识别并处理敏感信息（邮箱/信用卡/IP 等），
     策略有 redact(替换占位)/mask(打码)/hash(哈希)/block(拦截)。
  2) ModelCallLimitMiddleware：限制单次运行的模型调用次数，
     防止 Agent 陷入死循环、控制成本。

运行: python demo_06_guardrails_pii.py
"""

from _bootstrap import setup_path

setup_path()

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, PIIMiddleware

from smart_assistant.config import build_model, has_valid_key
from smart_assistant.tools import SAFE_TOOLS


def main():
    if not has_valid_key():
        print("[!] 未配置有效 DASHSCOPE_API_KEY，无法实跑。可先阅读代码理解结构。")
        return

    agent = create_agent(
        model=build_model(),
        tools=SAFE_TOOLS,
        system_prompt="你是智能助手。",
        middleware=[
            # 邮箱脱敏：用户输入里的邮箱会被替换，不会原样进入模型
            PIIMiddleware("email", strategy="redact", apply_to_input=True),
            # 调用限制：单次运行最多 5 次模型调用，超了优雅结束
            ModelCallLimitMiddleware(run_limit=5, exit_behavior="end"),
        ],
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "我的邮箱是 zhangsan@test.com，帮我记一下，然后查北京天气。"}]}
    )
    print("助手:", result["messages"][-1].content)
    print("\n(PII 护栏会在邮箱进入模型前将其 redact，保护隐私)")


if __name__ == "__main__":
    main()

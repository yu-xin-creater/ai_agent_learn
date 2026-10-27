"""
Demo 08 - 结构化输出（Structured Output）
补充时间: 2026-07-23

要点：生产里常需要 Agent 返回"可被程序消费"的结构化数据(JSON)，而非自由文本。
      1.0 用 create_agent(response_format=你的Pydantic模型)，
      结果可在 result["structured_response"] 取到类型化对象。

应用场景：信息抽取、意图分类、表单填充、把 Agent 接进下游系统。

运行: python demo_08_structured_output.py
"""

from _bootstrap import setup_path

setup_path()

from pydantic import BaseModel, Field

from langchain.agents import create_agent

from smart_assistant.config import build_model, has_valid_key
from smart_assistant.tools import SAFE_TOOLS


# 定义期望的输出结构
class WeatherReport(BaseModel):
    """结构化的天气报告"""

    city: str = Field(description="城市名")
    condition: str = Field(description="天气状况，如晴/多云/雨")
    summary: str = Field(description="一句话总结，是否适合外出")


def main():
    if not has_valid_key():
        print("[!] 未配置有效 DASHSCOPE_API_KEY，无法实跑。可先阅读代码理解结构。")
        return

    agent = create_agent(
        model=build_model(),
        tools=SAFE_TOOLS,
        system_prompt="你是天气助手，先用工具查天气，再按要求的结构输出。",
        response_format=WeatherReport,   # 指定结构化输出
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "查一下广州的天气并给出结构化报告"}]}
    )

    # 结构化结果在 structured_response 中，是 WeatherReport 实例
    report: WeatherReport = result["structured_response"]
    print("结构化输出对象:", report)
    print(f"  城市: {report.city}")
    print(f"  天气: {report.condition}")
    print(f"  总结: {report.summary}")


if __name__ == "__main__":
    main()

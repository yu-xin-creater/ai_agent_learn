"""
Demo 07 - 流式输出（Streaming）
补充时间: 2026-07-23

要点：生产中用户不该干等几十秒。agent.stream(...) 可边生成边返回，
      显著改善体验。stream_mode 可选：
        "values"   每步后返回完整状态
        "updates"  每步返回增量更新
        "messages" 逐 token 流式返回模型输出（最适合打字机效果）

运行: python demo_07_streaming.py
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

    agent = create_agent(
        model=build_model(),
        tools=SAFE_TOOLS,
        system_prompt=SYSTEM_PROMPT,
    )

    print("助手(流式): ", end="", flush=True)
    # stream_mode="messages" 逐 token 返回，实现打字机效果
    for token, metadata in agent.stream(
        {"messages": [{"role": "user", "content": "用两句话介绍一下杭州这座城市。"}]},
        stream_mode="messages",
    ):
        # token 是消息块，content 为增量文本
        content = getattr(token, "content", "")
        if isinstance(content, str) and content:
            print(content, end="", flush=True)
    print("\n")


if __name__ == "__main__":
    main()

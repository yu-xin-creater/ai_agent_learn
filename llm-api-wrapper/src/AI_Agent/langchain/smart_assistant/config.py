"""
配置模块（LangChain 1.0 版）

集中管理模型构建、API Key、LangSmith 可观测性开关。
补充时间: 2026-07-23
"""

import os

from dotenv import load_dotenv

load_dotenv()


# ============================================================================
# 模型配置
# ============================================================================
MODEL_NAME = os.getenv("ASSISTANT_MODEL", "qwen-plus")
TEMPERATURE = float(os.getenv("ASSISTANT_TEMPERATURE", "0.7"))


def build_model(temperature: float = None, model_name: str = None):
    """
    构建聊天模型（通义千问）

    为什么单独封装：
    - 多个 demo / agent 共用同一处模型配置，改一处即可全局生效
    - 方便切换厂商（下面给了 OpenAI / Anthropic 的写法注释）

    Args:
        temperature: 采样温度，覆盖默认值
        model_name: 模型名，覆盖默认值

    Returns:
        BaseChatModel 实例
    """
    from langchain_community.chat_models import ChatTongyi

    return ChatTongyi(
        model=model_name or MODEL_NAME,
        temperature=temperature if temperature is not None else TEMPERATURE,
        dashscope_api_key=os.environ.get("DASHSCOPE_API_KEY"),
    )

    # —— 切换到其他厂商的写法（1.0 下接口统一，换个类即可）——
    # from langchain_openai import ChatOpenAI
    # return ChatOpenAI(model="gpt-4o-mini", temperature=...)
    #
    # from langchain_anthropic import ChatAnthropic
    # return ChatAnthropic(model="claude-sonnet-5", temperature=...)


def has_valid_key() -> bool:
    """检查是否配置了有效的 API Key（排除占位符/过期占位）"""
    key = os.environ.get("DASHSCOPE_API_KEY", "")
    placeholders = {"", "your_api_key_here", "sk-xxx"}
    return key not in placeholders


# ============================================================================
# LangSmith 可观测性（企业生产必备：追踪每一步调用）
# ============================================================================
def enable_langsmith(project: str = "smart-assistant"):
    """
    开启 LangSmith 追踪

    开启后在 https://smith.langchain.com 可查看每次 agent 运行的
    完整调用链：模型输入输出、工具调用、token、延迟等。

    使用前需在 .env 设置:
        LANGSMITH_API_KEY=ls__xxx
    """
    api_key = os.environ.get("LANGSMITH_API_KEY")
    if not api_key:
        print("[LangSmith] 未设置 LANGSMITH_API_KEY，跳过追踪")
        return False

    # LangChain 1.0 通过环境变量启用追踪
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_PROJECT"] = project
    print(f"[LangSmith] 已开启追踪，项目: {project}")
    return True


# ============================================================================
# 统一的系统提示词
# ============================================================================
SYSTEM_PROMPT = """你是小智，一个全能智能助手。

你可以帮用户：
• 查询天气：查询中国主要城市的天气
• 数学计算：进行各种数学运算
• 文本翻译：将中文翻译成其他语言
• 知识查询：查询退货、配送、发票等公司 FAQ
• 发送邮件 / 提交订单（敏感操作，会先请求确认）

工作原则：
1. 先理解用户意图，选择合适的工具
2. 不确定时主动询问用户
3. 回答简洁、准确、有帮助
"""

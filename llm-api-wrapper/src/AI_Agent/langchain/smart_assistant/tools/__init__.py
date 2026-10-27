"""
工具集合

统一导出所有工具，方便 agent_factory 一次性导入。
新增工具时在这里登记即可。
"""

from .actions import place_order, send_email
from .calculator import calculator
from .knowledge import search_knowledge
from .translator import translate
from .weather import get_weather

# 普通工具（无副作用，可自由调用）
SAFE_TOOLS = [get_weather, calculator, translate, search_knowledge]

# 敏感工具（有副作用，建议配合人工审批）
SENSITIVE_TOOLS = [send_email, place_order]

# 全部工具
ALL_TOOLS = SAFE_TOOLS + SENSITIVE_TOOLS

__all__ = [
    "get_weather",
    "calculator",
    "translate",
    "search_knowledge",
    "send_email",
    "place_order",
    "SAFE_TOOLS",
    "SENSITIVE_TOOLS",
    "ALL_TOOLS",
]

"""
自定义 Middleware 示例（LangChain 1.0 核心概念）
补充时间: 2026-07-23

════════════════════════════════════════════════════════════════════
Middleware 是 LangChain 1.0 的灵魂。
它可组合地"插"进 agent 的执行循环（模型调用、工具调用的前后），
用来控制：动态提示词、对话摘要、工具权限、护栏、状态管理等。

Agent 循环示意：
    用户输入
      │
      ▼
  [before_model]  ← 可在调模型前改状态/拦截
      │
      ▼
   模型调用  ←  [dynamic_prompt 在这里动态生成系统提示词]
      │
      ▼
  [after_model]   ← 可在模型返回后检查/改写
      │
      ├── 需要调工具？→ 工具调用 → 回到 before_model
      └── 不调工具 → 结束
════════════════════════════════════════════════════════════════════

本文件用三个装饰器演示自定义 middleware：
  @dynamic_prompt  动态系统提示词
  @before_model    模型调用前钩子
  @after_model     模型调用后钩子
"""

from datetime import datetime

from langchain.agents.middleware import (
    after_model,
    before_model,
    dynamic_prompt,
)

from .config import SYSTEM_PROMPT


# ============================================================================
# 1. 动态提示词 Middleware
# ----------------------------------------------------------------------------
# 作用：每次调用模型前，动态拼接系统提示词（注入实时信息，如当前时间）。
#
# 【新旧对照】
#   0.x：系统提示词是写死的字符串，想注入动态信息要自己拼 messages，很笨拙
#   1.0：用 @dynamic_prompt，函数每次调用模型前执行，返回最终 system prompt
# ============================================================================
@dynamic_prompt
def time_aware_prompt(request) -> str:
    """在系统提示词里动态注入当前时间，让助手具备时间感知"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S %A")
    return f"{SYSTEM_PROMPT}\n\n【当前时间】{now}（回答时间相关问题时请参考）"


# ============================================================================
# 2. before_model Middleware
# ----------------------------------------------------------------------------
# 作用：模型调用前执行。这里演示"轻量日志/审计"——打印即将处理的消息数。
#
# 【新旧对照】
#   0.x：要加这类钩子得用 Callbacks 回调系统，概念分散、不好组合
#   1.0：@before_model 直接拿到 state，返回 dict 可更新状态（返回 None 则不改）
# ============================================================================
@before_model
def audit_before_model(state, runtime):
    """模型调用前的审计日志（生产中可换成写入日志系统）"""
    messages = state.get("messages", [])
    print(f"  [middleware:before_model] 即将调用模型，当前消息数={len(messages)}")
    return None  # 返回 None 表示不修改状态


# ============================================================================
# 3. after_model Middleware
# ----------------------------------------------------------------------------
# 作用：模型返回后执行。这里演示记录模型产出的字符数（可用于成本/质量监控）。
# ============================================================================
@after_model
def audit_after_model(state, runtime):
    """模型返回后的审计日志"""
    messages = state.get("messages", [])
    if messages:
        last = messages[-1]
        content = getattr(last, "content", "")
        length = len(content) if isinstance(content, str) else 0
        print(f"  [middleware:after_model] 模型已返回，本次输出约 {length} 字")
    return None


# 导出：按需挑选组合
CUSTOM_MIDDLEWARE = [time_aware_prompt, audit_before_model, audit_after_model]

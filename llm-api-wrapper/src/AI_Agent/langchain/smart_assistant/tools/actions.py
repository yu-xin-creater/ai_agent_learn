"""敏感操作工具（模拟实现）

这类"会产生副作用"的工具（发邮件、下单、转账等）是
Human-in-the-Loop（人工介入）middleware 的典型保护对象：
执行前应暂停、等人工确认，避免 Agent 误操作。
"""

from langchain_core.tools import tool


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """
    发送一封邮件（敏感操作，建议配合人工审批使用）。

    Args:
        to: 收件人邮箱
        subject: 邮件主题
        body: 邮件正文
    """
    # 真实实现会调用邮件服务；这里仅模拟
    return f"[已发送] 收件人={to} 主题={subject} 正文长度={len(body)}字"


@tool
def place_order(product: str, quantity: int) -> str:
    """
    提交一个采购订单（敏感操作，建议配合人工审批使用）。

    Args:
        product: 商品名称
        quantity: 数量
    """
    return f"[已下单] 商品={product} 数量={quantity}"

"""知识库查询工具（模拟实现）

演示用：一个简单的内置 FAQ 查询。
真实项目中这里应接入前面 RAG 模块的检索能力（向量库 / API）。
"""

from langchain_core.tools import tool


# 模拟的知识库（真实场景应替换为向量检索）
_KNOWLEDGE_BASE = {
    "退货": "本店支持 7 天无理由退货，商品需保持完好，联系客服发起即可。",
    "配送": "下单后 48 小时内发货，一般 3-5 个工作日送达，偏远地区顺延。",
    "发票": "支持开具电子发票，在订单详情页点击“申请发票”填写抬头即可。",
    "会员": "累计消费满 1000 元自动升级为 VIP 会员，享受 9 折优惠和专属客服。",
    "营业时间": "在线客服工作时间为每天 9:00-21:00，其余时间可留言。",
}


@tool
def search_knowledge(keyword: str) -> str:
    """
    从公司知识库/FAQ 中查询信息，适用于退货、配送、发票、会员、营业时间等问题。

    Args:
        keyword: 查询关键词，如 "退货"、"配送"、"发票"
    """
    # 简单的包含匹配（真实项目请用向量语义检索）
    hits = [v for k, v in _KNOWLEDGE_BASE.items() if k in keyword or keyword in k]
    if hits:
        return "\n".join(hits)
    return f"知识库中暂无与“{keyword}”相关的信息，可咨询在线客服。"

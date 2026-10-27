"""
示例: 从后台 API 拉取数据并灌入 RAG 知识库

演示两种入库方式:
  1. rebuild=True  清空并用 API 数据重建知识库
  2. rebuild=False 在现有知识库基础上增量追加

运行:
    python src/AI_Agent/RAG/demo_api_ingest.py
    或 python -m src.AI_Agent.RAG.demo_api_ingest
"""

import os
import sys

if __package__ in (None, ""):
    _PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PARENT_DIR not in sys.path:
        sys.path.insert(0, _PARENT_DIR)
    import RAG  # noqa: F401
    __package__ = "RAG"

from .api_loader import APILoader, ingest_from_api
from .config import default_config
from .rag_pipeline import RAGPipeline


def demo_load_only():
    """只演示"拉取 + 解析"，不消耗 Embedding 费用"""
    print("=" * 60)
    print("演示 1: 从公开 API 拉取数据并解析为 Document")
    print("=" * 60)

    loader = APILoader(timeout=30)
    # jsonplaceholder 是公开测试接口，返回 [{userId,id,title,body}, ...]
    docs = loader.load(
        url="https://jsonplaceholder.typicode.com/posts",
        params={"_limit": 5},           # 只取 5 条
        records_path=None,              # 顶层就是 list，自动识别
        content_fields=["title", "body"],
        metadata_fields=["id", "userId"],
    )

    print(f"\n共解析出 {len(docs)} 个 Document:\n")
    for i, doc in enumerate(docs, 1):
        preview = doc.content.replace("\n", " ")[:70]
        print(f"[{i}] metadata={doc.metadata}")
        print(f"    content: {preview}...\n")
    return docs


def demo_ingest_to_kb():
    """完整演示: API 数据入库并问答（需要有效的 Embedding/LLM Key）"""
    print("=" * 60)
    print("演示 2: 将 API 数据灌入知识库并问答")
    print("=" * 60)

    if default_config.llm.api_key in ("", "your_api_key_here"):
        print("[跳过] 未配置有效 API Key，无法向量化。请先在 .env 中设置。")
        return

    pipeline = RAGPipeline(config=default_config)

    # 一步灌入（rebuild=True 会用 API 数据重建知识库）
    count = ingest_from_api(
        pipeline,
        url="https://jsonplaceholder.typicode.com/posts",
        rebuild=True,
        params={"_limit": 10},
        content_fields=["title", "body"],
        metadata_fields=["id", "userId"],
    )
    print(f"\n入库完成，共 {count} 个文本块\n")

    # 问答验证
    response = pipeline.query("这些文章大致在讲什么内容?")
    print(response.format_answer())


if __name__ == "__main__":
    docs = demo_load_only()
    print()
    # 如需完整入库+问答，取消下面注释
    # demo_ingest_to_kb()

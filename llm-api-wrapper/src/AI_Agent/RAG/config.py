"""
RAG 系统全局配置模块

集中管理所有可配置参数，方便统一调整和扩展。
使用 dataclass 组织配置，支持从环境变量或配置文件加载。
"""

import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

# 加载 .env 文件中的环境变量
load_dotenv()


# ============================================================================
# LLM 模型配置
# ============================================================================
@dataclass
class LLMConfig:
    """大语言模型配置"""

    # API 密钥（从环境变量读取，安全起见不硬编码）
    api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    # API 基础地址（支持 OpenAI 兼容接口）
    base_url: str = field(
        default_factory=lambda: os.getenv(
            "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    )
    # 模型名称
    model_name: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "qwen-plus")
    )
    # 生成温度（0~1，越低越确定性）
    temperature: float = 0.7
    # 最大生成 token 数
    max_tokens: int = 2048
    # Top-P 采样
    top_p: float = 0.9
    # 请求超时时间（秒）
    timeout: int = 60


# ============================================================================
# Embedding 模型配置
# ============================================================================
@dataclass
class EmbeddingConfig:
    """嵌入模型配置"""

    # Embedding API 密钥（可复用 LLM 的 key）
    api_key: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_API_KEY", os.getenv("LLM_API_KEY", "")
        )
    )
    # Embedding API 基础地址
    base_url: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    )
    # Embedding 模型名称
    model_name: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL", "text-embedding-v3"
        )
    )
    # 嵌入向量维度（需与模型匹配）
    dimension: int = 1024
    # 批量处理大小
    # 注意: 通义千问 text-embedding-v3 单次请求最多 10 条，超过会报 InvalidParameter
    batch_size: int = 10


# ============================================================================
# 文本分割配置
# ============================================================================
@dataclass
class SplitterConfig:
    """文本分割器配置"""

    # 每个块的最大字符数
    chunk_size: int = 500
    # 相邻块之间的重叠字符数（保证上下文连贯性）
    chunk_overlap: int = 100
    # 分割策略: "recursive" | "semantic" | "sentence"
    strategy: str = "recursive"
    # 分隔符优先级列表（递归分割时使用）
    separators: list = field(
        default_factory=lambda: ["\n\n", "\n", "。", "！", "？", ".", "!", "?", " "]
    )


# ============================================================================
# 向量存储配置
# ============================================================================
@dataclass
class VectorStoreConfig:
    """向量存储配置"""

    # 向量库类型: "faiss" | "chroma"
    store_type: str = "faiss"
    # 持久化存储路径
    persist_directory: str = field(
        default_factory=lambda: os.path.join(
            os.path.dirname(__file__), "vector_db"
        )
    )
    # 集合名称（Chroma 使用）
    collection_name: str = "rag_knowledge_base"
    # FAISS 索引类型: "flat" | "ivf" | "hnsw"
    faiss_index_type: str = "flat"


# ============================================================================
# 检索器配置
# ============================================================================
@dataclass
class RetrieverConfig:
    """检索器配置"""

    # 返回的最相关文档数量
    top_k: int = 5
    # 相似度阈值（低于此值的结果会被过滤）
    score_threshold: float = 0.3
    # 检索策略: "similarity" | "mmr" | "hybrid"
    # - similarity: 纯向量相似度检索
    # - mmr: 最大边际相关性（兼顾相关性和多样性）
    # - hybrid: 混合检索（向量 + 关键词）
    search_type: str = "similarity"
    # MMR 多样性参数（0=最大多样性，1=最大相关性）
    mmr_lambda: float = 0.7
    # 是否启用查询改写
    enable_query_rewrite: bool = False
    # 是否启用重排序
    enable_rerank: bool = False


# ============================================================================
# RAG 管道配置
# ============================================================================
@dataclass
class RAGConfig:
    """RAG 管道整体配置（聚合所有子配置）"""

    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    splitter: SplitterConfig = field(default_factory=SplitterConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)

    # 知识库文档目录
    knowledge_base_dir: str = field(
        default_factory=lambda: os.path.join(
            os.path.dirname(__file__), "knowledge_base"
        )
    )
    # 是否开启调试模式（打印中间过程）
    debug: bool = True


# 创建默认全局配置实例
default_config = RAGConfig()

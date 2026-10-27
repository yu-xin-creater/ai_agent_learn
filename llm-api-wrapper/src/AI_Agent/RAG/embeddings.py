"""
嵌入模型模块

将文本转换为高维向量表示（Embedding），用于语义相似度计算。
支持多种嵌入方式，包括在线 API 和本地模型。

学习要点:
- Embedding 的作用: 将文本映射到向量空间，语义相近的文本距离更近
- 在线 vs 本地模型的取舍:
  - 在线 API: 效果好、无需 GPU，但有网络延迟和成本
  - 本地模型: 免费、低延迟，但需要计算资源
- 批量处理的重要性: 减少 API 调用次数
"""

import hashlib
import json
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from .config import EmbeddingConfig


# ============================================================================
# 嵌入模型基类
# ============================================================================
class BaseEmbedding(ABC):
    """嵌入模型抽象基类"""

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """
        将单条文本转换为向量

        Args:
            text: 输入文本

        Returns:
            浮点数向量
        """
        pass

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        批量将文本转换为向量

        Args:
            texts: 输入文本列表

        Returns:
            向量列表
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """返回向量维度"""
        pass


# ============================================================================
# OpenAI 兼容 API 嵌入模型（推荐）
# ============================================================================
class OpenAICompatibleEmbedding(BaseEmbedding):
    """
    OpenAI 兼容接口的嵌入模型

    支持所有兼容 OpenAI Embedding API 的服务:
    - 通义千问 text-embedding-v3
    - OpenAI text-embedding-3-small/large
    - DeepSeek embedding
    - 其他兼容接口
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        if config is None:
            from .config import default_config
            config = default_config.embedding

        self.config = config
        self._dimension = config.dimension

        # 使用 openai 客户端（兼容接口）
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("需要安装 openai: pip install openai")

        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_text(self, text: str) -> List[float]:
        """调用 API 获取单条文本的向量"""
        response = self.client.embeddings.create(
            model=self.config.model_name,
            input=text,
        )
        return response.data[0].embedding

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        批量获取向量

        自动按 batch_size 分批处理，避免超出 API 限制。
        """
        all_embeddings = []
        batch_size = self.config.batch_size

        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            response = self.client.embeddings.create(
                model=self.config.model_name,
                input=batch,
            )
            # 按原始顺序排列结果
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings


# ============================================================================
# 本地嵌入模型（使用 sentence-transformers）
# ============================================================================
class LocalEmbedding(BaseEmbedding):
    """
    本地嵌入模型

    使用 sentence-transformers 加载本地模型，无需网络请求。
    推荐模型:
    - BAAI/bge-small-zh-v1.5 (中文，小型)
    - BAAI/bge-base-zh-v1.5 (中文，基础)
    - sentence-transformers/all-MiniLM-L6-v2 (英文，轻量)
    """

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "使用本地嵌入需要安装: pip install sentence-transformers"
            )

        self.model = SentenceTransformer(model_name)
        self._dimension = self.model.get_sentence_embedding_dimension()
        self.model_name = model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_text(self, text: str) -> List[float]:
        """本地模型编码单条文本"""
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """本地模型批量编码"""
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()


# ============================================================================
# 带缓存的嵌入模型（进阶优化：减少重复计算）
# ============================================================================
class CachedEmbedding(BaseEmbedding):
    """
    带本地缓存的嵌入模型装饰器

    对重复文本的嵌入结果进行缓存，避免重复调用 API，
    节省费用和时间。使用文本的 MD5 作为缓存键。

    这是装饰器模式的应用。
    """

    def __init__(
        self,
        base_embedding: BaseEmbedding,
        cache_dir: Optional[str] = None,
    ):
        """
        Args:
            base_embedding: 被装饰的嵌入模型
            cache_dir: 缓存文件存储目录
        """
        self.base = base_embedding
        self.cache_dir = cache_dir or os.path.join(
            os.path.dirname(__file__), ".embedding_cache"
        )
        os.makedirs(self.cache_dir, exist_ok=True)
        # 内存缓存
        self._memory_cache: Dict[str, List[float]] = {}

    @property
    def dimension(self) -> int:
        return self.base.dimension

    def _get_cache_key(self, text: str) -> str:
        """生成缓存键（文本的 MD5）"""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _get_from_cache(self, text: str) -> Optional[List[float]]:
        """尝试从缓存获取"""
        key = self._get_cache_key(text)

        # 先查内存缓存
        if key in self._memory_cache:
            return self._memory_cache[key]

        # 再查文件缓存
        cache_file = os.path.join(self.cache_dir, f"{key}.json")
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                embedding = json.load(f)
                self._memory_cache[key] = embedding
                return embedding

        return None

    def _save_to_cache(self, text: str, embedding: List[float]):
        """保存到缓存"""
        key = self._get_cache_key(text)
        self._memory_cache[key] = embedding
        cache_file = os.path.join(self.cache_dir, f"{key}.json")
        with open(cache_file, "w") as f:
            json.dump(embedding, f)

    def embed_text(self, text: str) -> List[float]:
        """带缓存的单条嵌入"""
        cached = self._get_from_cache(text)
        if cached is not None:
            return cached

        embedding = self.base.embed_text(text)
        self._save_to_cache(text, embedding)
        return embedding

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """带缓存的批量嵌入"""
        results = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []

        # 检查哪些已缓存
        for i, text in enumerate(texts):
            cached = self._get_from_cache(text)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        # 批量处理未缓存的
        if uncached_texts:
            new_embeddings = self.base.embed_texts(uncached_texts)
            for idx, embedding in zip(uncached_indices, new_embeddings):
                results[idx] = embedding
                self._save_to_cache(texts[idx], embedding)

        return results


# ============================================================================
# 嵌入模型工厂
# ============================================================================
def create_embedding(
    provider: str = "openai_compatible",
    config: Optional[EmbeddingConfig] = None,
    use_cache: bool = True,
) -> BaseEmbedding:
    """
    创建嵌入模型实例

    Args:
        provider: 提供者类型 "openai_compatible" | "local"
        config: 嵌入配置
        use_cache: 是否启用缓存

    Returns:
        嵌入模型实例
    """
    if provider == "openai_compatible":
        embedding = OpenAICompatibleEmbedding(config)
    elif provider == "local":
        model_name = config.model_name if config else "BAAI/bge-small-zh-v1.5"
        embedding = LocalEmbedding(model_name)
    else:
        raise ValueError(f"未知的嵌入模型提供者: {provider}")

    # 装饰缓存层
    if use_cache:
        embedding = CachedEmbedding(embedding)

    return embedding

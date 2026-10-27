"""
向量存储模块

管理文档向量的存储、索引和检索。
支持 FAISS（轻量高效）和 Chroma（功能丰富）两种后端。

学习要点:
- 向量数据库的核心功能: 存储、索引、检索
- FAISS vs Chroma:
  - FAISS: Facebook 开源，速度极快，适合大规模数据
  - Chroma: API 友好，支持元数据过滤，适合原型开发
- 索引类型:
  - Flat: 暴力搜索，精确但慢（适合小数据集）
  - IVF: 倒排索引，快但近似（适合中等数据集）
  - HNSW: 图索引，快且准（适合大数据集）
- 持久化: 将索引保存到磁盘，避免重复构建
"""

import os
import pickle
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import VectorStoreConfig
from .document_loader import Document
from .embeddings import BaseEmbedding


# ============================================================================
# 检索结果数据结构
# ============================================================================
class SearchResult:
    """
    检索结果

    Attributes:
        document: 匹配的文档
        score: 相似度得分（越高越相关）
        rank: 排名位置
    """

    def __init__(self, document: Document, score: float, rank: int = 0):
        self.document = document
        self.score = score
        self.rank = rank

    def __repr__(self) -> str:
        preview = self.document.content[:60] + "..."
        return f"SearchResult(score={self.score:.4f}, content='{preview}')"


# ============================================================================
# 向量存储基类
# ============================================================================
class BaseVectorStore(ABC):
    """向量存储抽象基类"""

    @abstractmethod
    def add_documents(
        self, documents: List[Document], embeddings: List[List[float]]
    ):
        """添加文档及其向量到存储"""
        pass

    @abstractmethod
    def search(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[SearchResult]:
        """根据查询向量检索最相似的文档"""
        pass

    @abstractmethod
    def reset(self):
        """清空存储中的所有向量与文档"""
        pass

    @abstractmethod
    def save(self, path: str):
        """持久化存储到磁盘"""
        pass

    @abstractmethod
    def load(self, path: str):
        """从磁盘加载存储"""
        pass

    @property
    @abstractmethod
    def count(self) -> int:
        """返回存储的文档数量"""
        pass


# ============================================================================
# FAISS 向量存储
# ============================================================================
class FAISSVectorStore(BaseVectorStore):
    """
    基于 FAISS 的向量存储

    FAISS (Facebook AI Similarity Search) 是高效的向量相似度搜索库。
    支持多种索引类型:
    - flat: L2 距离暴力搜索（精确，适合 <10w 数据）
    - ivf: 倒排文件索引（快速，适合 10w~100w 数据）
    - hnsw: 层次可导航小世界图（综合最优）
    """

    def __init__(
        self,
        dimension: int,
        index_type: str = "flat",
        config: Optional[VectorStoreConfig] = None,
    ):
        """
        Args:
            dimension: 向量维度
            index_type: 索引类型 "flat" | "ivf" | "hnsw"
            config: 向量存储配置
        """
        try:
            import faiss
        except ImportError:
            raise ImportError("需要安装 faiss: pip install faiss-cpu")

        self.dimension = dimension
        self.index_type = index_type
        self.config = config
        self.documents: List[Document] = []

        # 创建 FAISS 索引
        self.index = self._create_index(dimension, index_type)

    def _create_index(self, dimension: int, index_type: str):
        """根据类型创建 FAISS 索引"""
        import faiss

        if index_type == "flat":
            # 精确的 L2 距离搜索（暴力搜索）
            index = faiss.IndexFlatIP(dimension)  # 内积（用于归一化向量=余弦相似度）
        elif index_type == "ivf":
            # 倒排文件索引（近似搜索，需要训练）
            nlist = 100  # 聚类中心数
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        elif index_type == "hnsw":
            # HNSW 图索引
            index = faiss.IndexHNSWFlat(dimension, 32)  # 32 为连接数
            index.hnsw.efConstruction = 200  # 构建时搜索范围
            index.hnsw.efSearch = 64  # 搜索时搜索范围
        else:
            raise ValueError(f"不支持的索引类型: {index_type}")

        return index

    def add_documents(
        self, documents: List[Document], embeddings: List[List[float]]
    ):
        """添加文档到 FAISS 索引"""
        import faiss

        vectors = np.array(embeddings, dtype=np.float32)

        # 归一化向量（使内积等价于余弦相似度）
        faiss.normalize_L2(vectors)

        # IVF 索引需要先训练
        if self.index_type == "ivf" and not self.index.is_trained:
            if len(vectors) >= 100:
                self.index.train(vectors)
            else:
                # 数据量不足时退回 Flat 索引
                self.index = self._create_index(self.dimension, "flat")

        self.index.add(vectors)
        self.documents.extend(documents)

    def reset(self):
        """
        清空索引与文档

        重新构建知识库前调用，避免向已有索引重复追加向量，
        导致 FAISS 索引与文档列表数量不一致。
        """
        self.index = self._create_index(self.dimension, self.index_type)
        self.documents = []

    def search(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[SearchResult]:
        """在 FAISS 索引中搜索"""
        import faiss

        if self.count == 0:
            return []

        # 转换并归一化查询向量
        query_vector = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_vector)

        # 搜索
        top_k = min(top_k, self.count)
        scores, indices = self.index.search(query_vector, top_k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            # FAISS 用 -1 表示无效结果；同时防护索引越界
            # （当持久化的 index 与 documents 数量不一致时，idx 可能超出范围）
            if idx < 0 or idx >= len(self.documents):
                continue
            results.append(
                SearchResult(
                    document=self.documents[idx],
                    score=float(score),
                    rank=rank,
                )
            )
        return results

    def save(self, path: Optional[str] = None):
        """保存索引和文档到磁盘"""
        import faiss

        save_dir = path or (self.config.persist_directory if self.config else "vector_db")
        os.makedirs(save_dir, exist_ok=True)

        # 保存 FAISS 索引
        index_path = os.path.join(save_dir, "faiss.index")
        faiss.write_index(self.index, index_path)

        # 保存文档列表
        docs_path = os.path.join(save_dir, "documents.pkl")
        with open(docs_path, "wb") as f:
            pickle.dump(self.documents, f)

        print(f"  [+] 向量索引已保存到: {save_dir}")

    def load(self, path: Optional[str] = None):
        """从磁盘加载索引和文档"""
        import faiss

        load_dir = path or (self.config.persist_directory if self.config else "vector_db")

        index_path = os.path.join(load_dir, "faiss.index")
        docs_path = os.path.join(load_dir, "documents.pkl")

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"索引文件不存在: {index_path}")

        self.index = faiss.read_index(index_path)
        with open(docs_path, "rb") as f:
            self.documents = pickle.load(f)

        print(f"  [+] 已加载向量索引，文档数: {self.count}")

    @property
    def count(self) -> int:
        return len(self.documents)


# ============================================================================
# Chroma 向量存储
# ============================================================================
class ChromaVectorStore(BaseVectorStore):
    """
    基于 Chroma 的向量存储

    Chroma 是一个开源的向量数据库，特点:
    - API 简洁，开箱即用
    - 支持元数据过滤
    - 自动持久化
    - 支持多种距离度量
    """

    def __init__(self, config: Optional[VectorStoreConfig] = None):
        try:
            import chromadb
        except ImportError:
            raise ImportError("需要安装 chromadb: pip install chromadb")

        self.config = config
        persist_dir = config.persist_directory if config else "vector_db/chroma"
        collection_name = config.collection_name if config else "rag_knowledge_base"

        # 创建/连接 Chroma 客户端
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # 使用余弦距离
        )
        self.documents: List[Document] = []

    def add_documents(
        self, documents: List[Document], embeddings: List[List[float]]
    ):
        """添加文档到 Chroma"""
        ids = [f"doc_{self.count + i}" for i in range(len(documents))]

        # Chroma 要求 metadata 值为基本类型
        metadatas = []
        for doc in documents:
            meta = {}
            for k, v in doc.metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    meta[k] = v
            metadatas.append(meta)

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=[doc.content for doc in documents],
            metadatas=metadatas,
        )
        self.documents.extend(documents)

    def reset(self):
        """清空 Chroma 集合并重建"""
        collection_name = self.collection.name
        self.client.delete_collection(name=collection_name)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.documents = []

    def search(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[SearchResult]:
        """在 Chroma 中搜索"""
        if self.count == 0:
            return []

        top_k = min(top_k, self.count)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        search_results = []
        for rank, (doc_content, metadata, distance) in enumerate(
            zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ):
            # Chroma 返回的是距离，转换为相似度
            score = 1 - distance
            document = Document(content=doc_content, metadata=metadata)
            search_results.append(
                SearchResult(document=document, score=score, rank=rank)
            )

        return search_results

    def save(self, path: Optional[str] = None):
        """Chroma 自动持久化，此方法为兼容接口"""
        print("  [+] Chroma 已自动持久化")

    def load(self, path: Optional[str] = None):
        """Chroma 自动加载，此方法为兼容接口"""
        self.documents = []
        # 从 collection 重建文档列表
        all_data = self.collection.get()
        if all_data and all_data["documents"]:
            for content, metadata in zip(
                all_data["documents"], all_data["metadatas"]
            ):
                self.documents.append(
                    Document(content=content, metadata=metadata or {})
                )
        print(f"  [+] Chroma 已加载，文档数: {self.count}")

    @property
    def count(self) -> int:
        return self.collection.count()


# ============================================================================
# 向量存储工厂
# ============================================================================
def create_vector_store(
    embedding_dim: int,
    config: Optional[VectorStoreConfig] = None,
) -> BaseVectorStore:
    """
    创建向量存储实例

    Args:
        embedding_dim: 嵌入向量维度
        config: 向量存储配置

    Returns:
        向量存储实例
    """
    if config is None:
        from .config import default_config
        config = default_config.vector_store

    if config.store_type == "faiss":
        return FAISSVectorStore(
            dimension=embedding_dim,
            index_type=config.faiss_index_type,
            config=config,
        )
    elif config.store_type == "chroma":
        return ChromaVectorStore(config=config)
    else:
        raise ValueError(f"不支持的向量库类型: {config.store_type}")

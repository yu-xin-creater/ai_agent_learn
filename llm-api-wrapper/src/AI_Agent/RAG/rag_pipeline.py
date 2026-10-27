"""
RAG 管道模块

将文档加载、分割、向量化、检索、生成串联成完整的流水线。
提供统一的高层 API，一键完成知识库构建和问答。

学习要点:
- Pipeline 设计模式: 将复杂流程分解为独立步骤
- 两阶段工作流:
  1. 离线阶段(Indexing): 文档 → 分割 → 向量化 → 存储
  2. 在线阶段(Querying): 查询 → 检索 → 生成答案
- 可观测性: 通过 debug 模式查看每个步骤的中间结果
"""

import os
import time
from dataclasses import dataclass
from typing import Generator, List, Optional

from .config import RAGConfig, default_config
from .document_loader import Document, DocumentLoaderFactory
from .embeddings import BaseEmbedding, create_embedding
from .generator import RAGGenerator
from .retriever import Retriever
from .text_splitter import BaseTextSplitter, create_splitter
from .vector_store import BaseVectorStore, SearchResult, create_vector_store


# ============================================================================
# 查询结果数据结构
# ============================================================================
@dataclass
class RAGResponse:
    """
    RAG 查询响应

    Attributes:
        answer: LLM 生成的回答
        sources: 引用的源文档列表
        query: 原始查询
        elapsed_time: 耗时（秒）
    """

    answer: str
    sources: List[SearchResult]
    query: str
    elapsed_time: float = 0.0

    def __repr__(self) -> str:
        return (
            f"RAGResponse(\n"
            f"  query='{self.query}',\n"
            f"  answer='{self.answer[:100]}...',\n"
            f"  sources_count={len(self.sources)},\n"
            f"  elapsed_time={self.elapsed_time:.2f}s\n"
            f")"
        )

    def format_answer(self) -> str:
        """格式化输出完整回答（含来源信息）"""
        output = f"\n{'='*60}\n"
        output += f"问题: {self.query}\n"
        output += f"{'─'*60}\n"
        output += f"回答:\n{self.answer}\n"
        output += f"{'─'*60}\n"
        output += f"参考来源 ({len(self.sources)} 个文档片段):\n"
        for i, source in enumerate(self.sources, 1):
            filename = source.document.metadata.get("filename", "未知")
            score = source.score
            preview = source.document.content[:80].replace("\n", " ")
            output += f"  [{i}] {filename} (相关度: {score:.3f})\n"
            output += f"      {preview}...\n"
        output += f"{'─'*60}\n"
        output += f"耗时: {self.elapsed_time:.2f}s\n"
        output += f"{'='*60}\n"
        return output


# ============================================================================
# RAG 管道
# ============================================================================
class RAGPipeline:
    """
    RAG 管道主类

    提供完整的 RAG 工作流:
    1. build_knowledge_base() - 构建知识库（离线）
    2. query() - 知识问答（在线）
    3. query_stream() - 流式问答（在线）

    使用示例:
        pipeline = RAGPipeline()
        pipeline.build_knowledge_base("./knowledge_base")
        response = pipeline.query("什么是 RAG?")
        print(response.format_answer())
    """

    def __init__(self, config: Optional[RAGConfig] = None):
        """
        Args:
            config: RAG 配置，为 None 时使用默认配置
        """
        self.config = config or default_config
        self.debug = self.config.debug

        # 初始化各组件
        self._init_components()

    def _init_components(self):
        """初始化所有组件"""
        if self.debug:
            print("[RAG Pipeline] 正在初始化组件...")

        # 1. 文本分割器
        self.splitter: BaseTextSplitter = create_splitter(self.config.splitter)
        if self.debug:
            print(f"  [+] 分割器: {self.config.splitter.strategy}, "
                  f"chunk_size={self.config.splitter.chunk_size}")

        # 2. 嵌入模型
        self.embedding_model: BaseEmbedding = create_embedding(
            config=self.config.embedding
        )
        if self.debug:
            print(f"  [+] 嵌入模型: {self.config.embedding.model_name}")

        # 3. 向量存储
        self.vector_store: BaseVectorStore = create_vector_store(
            embedding_dim=self.config.embedding.dimension,
            config=self.config.vector_store,
        )
        if self.debug:
            print(f"  [+] 向量存储: {self.config.vector_store.store_type}")

        # 4. 生成器
        self.generator = RAGGenerator(config=self.config.llm)
        if self.debug:
            print(f"  [+] LLM 模型: {self.config.llm.model_name}")

        # 5. 检索器（需要生成器的 LLM 调用能力）
        self.retriever = Retriever(
            vector_store=self.vector_store,
            embedding_model=self.embedding_model,
            config=self.config.retriever,
            llm_call=self.generator.call_llm_simple,
        )
        if self.debug:
            print(f"  [+] 检索器: search_type={self.config.retriever.search_type}, "
                  f"top_k={self.config.retriever.top_k}")
            print("[RAG Pipeline] 初始化完成\n")

    # ========================================================================
    # 离线阶段: 知识库构建
    # ========================================================================
    def build_knowledge_base(
        self,
        source: Optional[str] = None,
        documents: Optional[List[Document]] = None,
        save: bool = True,
    ) -> int:
        """
        构建知识库（离线阶段）

        完整流程:
        文档加载 → 文本分割 → 向量化 → 存储

        Args:
            source: 知识库文档目录路径（与 documents 二选一）
            documents: 直接传入的文档列表
            save: 是否持久化存储

        Returns:
            处理的文档块数量
        """
        start_time = time.time()
        print(f"\n{'='*60}")
        print("[知识库构建] 开始...")
        print(f"{'='*60}")

        # Step 1: 加载文档
        if documents is None:
            source_dir = source or self.config.knowledge_base_dir
            print(f"\n[Step 1/4] 加载文档: {source_dir}")
            documents = DocumentLoaderFactory.load_directory(source_dir)
        else:
            print(f"\n[Step 1/4] 使用传入的文档: {len(documents)} 个")

        if not documents:
            print("[!] 未找到任何文档，请检查目录路径")
            return 0

        # Step 2: 文本分割
        print(f"\n[Step 2/4] 文本分割 (策略: {self.config.splitter.strategy})")
        chunks = self.splitter.split_documents(documents)
        print(f"  [+] 分割结果: {len(documents)} 个文档 → {len(chunks)} 个文本块")
        if self.debug and chunks:
            print(f"  [i] 平均块大小: {sum(len(c.content) for c in chunks) / len(chunks):.0f} 字符")

        # Step 3: 向量化
        print(f"\n[Step 3/4] 文本向量化 (模型: {self.config.embedding.model_name})")
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_model.embed_texts(texts)
        print(f"  [+] 已生成 {len(embeddings)} 个向量 (维度: {len(embeddings[0])})")

        # Step 4: 存储
        # 构建前先清空向量库，确保重复构建时索引与文档数量始终一致，
        # 避免向已有索引追加导致 FAISS 索引与 documents 错位。
        print(f"\n[Step 4/4] 存储到向量库 ({self.config.vector_store.store_type})")
        self.vector_store.reset()
        self.vector_store.add_documents(chunks, embeddings)

        # 持久化
        if save:
            self.vector_store.save()

        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"[知识库构建] 完成! 共处理 {len(chunks)} 个文本块, 耗时 {elapsed:.2f}s")
        print(f"{'='*60}\n")

        return len(chunks)

    def load_knowledge_base(self, path: Optional[str] = None):
        """
        加载已持久化的知识库

        Args:
            path: 向量库存储路径
        """
        print("[知识库] 正在加载已有向量库...")
        self.vector_store.load(path)
        print(f"[知识库] 加载完成，共 {self.vector_store.count} 个文档块\n")

    # ========================================================================
    # 在线阶段: 问答查询
    # ========================================================================
    def query(
        self,
        question: str,
        use_citation: bool = True,
        compress_context: bool = False,
    ) -> RAGResponse:
        """
        知识问答（在线阶段）

        完整流程:
        用户查询 → 检索相关文档 → 构建 Prompt → LLM 生成答案

        Args:
            question: 用户问题
            use_citation: 是否在回答中标注来源
            compress_context: 是否压缩上下文

        Returns:
            RAGResponse 对象（含答案、来源、耗时）
        """
        start_time = time.time()

        if self.debug:
            print(f"\n[查询] 问题: {question}")

        # Step 1: 检索
        if self.debug:
            print("[查询] Step 1 - 检索相关文档...")
        search_results = self.retriever.retrieve(question)

        if self.debug:
            print(f"  [+] 检索到 {len(search_results)} 个相关文档")
            for i, r in enumerate(search_results[:3], 1):
                print(f"      [{i}] score={r.score:.3f} | {r.document.content[:50]}...")

        # Step 2: 生成答案
        if self.debug:
            print("[查询] Step 2 - 生成答案...")
        answer = self.generator.generate(
            question=question,
            search_results=search_results,
            use_citation=use_citation,
            compress_context=compress_context,
        )

        elapsed = time.time() - start_time

        response = RAGResponse(
            answer=answer,
            sources=search_results,
            query=question,
            elapsed_time=elapsed,
        )

        if self.debug:
            print(f"[查询] 完成, 耗时 {elapsed:.2f}s")

        return response

    def query_stream(
        self, question: str
    ) -> Generator[str, None, None]:
        """
        流式问答

        适合需要实时显示生成过程的场景。

        Yields:
            每次生成的文本片段
        """
        # 检索
        search_results = self.retriever.retrieve(question)
        # 流式生成
        yield from self.generator.generate_stream(question, search_results)

    # ========================================================================
    # 工具方法
    # ========================================================================
    def add_document(self, text: str, metadata: Optional[dict] = None):
        """
        动态添加单个文档到知识库

        Args:
            text: 文档文本
            metadata: 元数据
        """
        doc = Document(content=text, metadata=metadata or {})
        self.add_documents([doc])

    def add_documents(
        self, documents: List[Document], save: bool = False
    ) -> int:
        """
        批量添加文档到已有知识库（增量，不清空原有数据）

        相比循环调用 add_document，本方法对所有文本块一次性批量向量化，
        效率更高。适合从外部 API / 数据库拉取的一批数据入库。

        Args:
            documents: 待添加的文档列表
            save: 是否在添加后立即持久化

        Returns:
            新增的文本块数量
        """
        if not documents:
            return 0
        chunks = self.splitter.split_documents(documents)
        texts = [c.content for c in chunks]
        embeddings = self.embedding_model.embed_texts(texts)
        self.vector_store.add_documents(chunks, embeddings)
        if save:
            self.vector_store.save()
        if self.debug:
            print(f"[+] 已批量添加 {len(documents)} 个文档 → {len(chunks)} 个文本块")
        return len(chunks)

    def clear_history(self):
        """清空对话历史"""
        self.generator.clear_history()

    @property
    def knowledge_base_size(self) -> int:
        """返回知识库中的文档块数量"""
        return self.vector_store.count

"""
文本分割模块

将长文档分割成适合向量化和检索的小块(chunks)。
提供多种分割策略，每种策略适用于不同的场景。

学习要点:
- 为什么需要分割: LLM 上下文长度限制 + 细粒度检索
- chunk_size 和 chunk_overlap 的权衡:
  - chunk_size 过大: 检索精度下降，噪声增多
  - chunk_size 过小: 上下文信息丢失
  - chunk_overlap: 确保块之间的语义连贯性
- 不同分割策略的适用场景
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from .config import SplitterConfig
from .document_loader import Document


# ============================================================================
# 分割器基类
# ============================================================================
class BaseTextSplitter(ABC):
    """文本分割器抽象基类"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        """
        Args:
            chunk_size: 每个文本块的最大字符数
            chunk_overlap: 相邻块之间的重叠字符数
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap 必须小于 chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """
        将文本分割成块

        Args:
            text: 待分割的文本

        Returns:
            分割后的文本块列表
        """
        pass

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        将文档列表中的每个文档分割成更小的文档

        Args:
            documents: 待分割的文档列表

        Returns:
            分割后的文档列表（保留并扩展元数据）
        """
        split_docs = []
        for doc in documents:
            chunks = self.split_text(doc.content)
            for idx, chunk in enumerate(chunks):
                # 继承原文档的元数据，并添加分割信息
                metadata = {
                    **doc.metadata,
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                    "chunk_size": len(chunk),
                }
                split_docs.append(Document(content=chunk, metadata=metadata))
        return split_docs


# ============================================================================
# 递归字符分割器（推荐默认使用）
# ============================================================================
class RecursiveCharacterSplitter(BaseTextSplitter):
    """
    递归字符分割器

    按照分隔符优先级列表递归分割文本。
    先尝试按段落分割，如果段落仍然过长，再按句子分割，
    如果句子仍然过长，最后按字符分割。

    这是最常用的分割策略，适合大多数场景。

    分割逻辑:
    1. 按优先级最高的分隔符分割文本
    2. 对于过长的片段，使用下一级分隔符继续分割
    3. 合并过短的相邻片段
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        separators: Optional[List[str]] = None,
    ):
        super().__init__(chunk_size, chunk_overlap)
        # 默认分隔符优先级：段落 > 换行 > 句号 > 空格 > 字符
        self.separators = separators or [
            "\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""
        ]

    def split_text(self, text: str) -> List[str]:
        """递归分割文本"""
        final_chunks = []
        self._split_recursive(text, self.separators, final_chunks)
        return final_chunks

    def _split_recursive(
        self, text: str, separators: List[str], chunks: List[str]
    ):
        """递归分割的核心逻辑"""
        if len(text) <= self.chunk_size:
            if text.strip():
                chunks.append(text.strip())
            return

        # 找到合适的分隔符
        separator = separators[-1]  # 默认使用最后一个（空字符=逐字符）
        for sep in separators:
            if sep == "":
                separator = sep
                break
            if sep in text:
                separator = sep
                break

        # 按分隔符分割
        if separator:
            splits = text.split(separator)
        else:
            # 空字符串分隔符 = 逐字符分割
            splits = list(text)

        # 合并分割后的片段，确保不超过 chunk_size
        current_chunk = ""
        for split in splits:
            piece = split + separator if separator else split
            if len(current_chunk) + len(piece) <= self.chunk_size:
                current_chunk += piece
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                # 保留重叠部分
                if self.chunk_overlap > 0 and current_chunk:
                    overlap = current_chunk[-self.chunk_overlap:]
                    current_chunk = overlap + piece
                else:
                    current_chunk = piece

                # 如果单个 piece 仍然过长，使用下一级分隔符
                if len(current_chunk) > self.chunk_size:
                    remaining_seps = separators[separators.index(separator) + 1:]
                    if remaining_seps:
                        self._split_recursive(current_chunk, remaining_seps, chunks)
                        current_chunk = ""

        # 处理最后一个块
        if current_chunk.strip():
            chunks.append(current_chunk.strip())


# ============================================================================
# 句子分割器
# ============================================================================
class SentenceSplitter(BaseTextSplitter):
    """
    基于句子的分割器

    先将文本按句子分割，再将句子组合成不超过 chunk_size 的块。
    适合需要保持句子完整性的场景。
    """

    # 中英文句子结束标点的正则
    SENTENCE_ENDINGS = re.compile(r'(?<=[。！？.!?])\s*')

    def split_text(self, text: str) -> List[str]:
        """按句子分割文本"""
        # 按句子分割
        sentences = self.SENTENCE_ENDINGS.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]

        # 将句子组合成块
        chunks = []
        current_chunk = ""
        current_sentences = []

        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= self.chunk_size:
                current_chunk += sentence + " "
                current_sentences.append(sentence)
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())

                # 计算重叠：保留最后几个句子作为重叠
                overlap_text = ""
                overlap_sentences = []
                for s in reversed(current_sentences):
                    if len(overlap_text) + len(s) <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_text = " ".join(overlap_sentences)
                    else:
                        break

                current_chunk = overlap_text + " " + sentence + " " if overlap_text else sentence + " "
                current_sentences = overlap_sentences + [sentence]

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks


# ============================================================================
# 固定大小分割器（最简单的方式）
# ============================================================================
class FixedSizeSplitter(BaseTextSplitter):
    """
    固定大小分割器

    简单地按固定字符数分割，不考虑语义边界。
    优点：实现简单、块大小一致
    缺点：可能在词语/句子中间断开
    适用于对语义完整性要求不高的场景。
    """

    def split_text(self, text: str) -> List[str]:
        """按固定大小分割"""
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            # 下一个块的起始位置 = 当前起始 + chunk_size - overlap
            start += self.chunk_size - self.chunk_overlap

        return chunks


# ============================================================================
# 分割器工厂
# ============================================================================
def create_splitter(config: Optional[SplitterConfig] = None) -> BaseTextSplitter:
    """
    根据配置创建分割器（工厂函数）

    Args:
        config: 分割器配置，为 None 时使用默认配置

    Returns:
        对应的分割器实例
    """
    if config is None:
        from .config import default_config
        config = default_config.splitter

    strategy_map = {
        "recursive": RecursiveCharacterSplitter,
        "sentence": SentenceSplitter,
        "fixed": FixedSizeSplitter,
    }

    splitter_class = strategy_map.get(config.strategy)
    if splitter_class is None:
        raise ValueError(
            f"未知的分割策略: {config.strategy}，"
            f"可选: {list(strategy_map.keys())}"
        )

    kwargs = {
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
    }
    # 递归分割器额外支持自定义分隔符
    if config.strategy == "recursive":
        kwargs["separators"] = config.separators

    return splitter_class(**kwargs)

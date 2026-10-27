"""
文档加载模块

负责从多种格式的文件中提取文本内容。
支持格式: .txt, .md, .pdf, .docx, .json
设计为可扩展的工厂模式，方便添加新的加载器。

学习要点:
- 工厂模式设计，通过文件后缀自动选择加载器
- 统一的 Document 数据结构
- 异常处理与日志记录
"""

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Type


# ============================================================================
# 文档数据结构
# ============================================================================
@dataclass
class Document:
    """
    统一的文档数据结构

    Attributes:
        content: 文档文本内容
        metadata: 元数据（来源文件、页码、创建时间等）
    """

    content: str
    metadata: Dict[str, any] = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.content[:80] + "..." if len(self.content) > 80 else self.content
        return f"Document(content='{preview}', metadata={self.metadata})"


# ============================================================================
# 加载器基类（抽象类）
# ============================================================================
class BaseLoader(ABC):
    """
    文档加载器抽象基类

    所有具体的加载器都需要继承此类并实现 load 方法。
    这是模板方法模式的应用。
    """

    @abstractmethod
    def load(self, file_path: str) -> List[Document]:
        """
        加载文件并返回文档列表

        Args:
            file_path: 文件路径

        Returns:
            Document 对象列表
        """
        pass

    def _create_metadata(self, file_path: str, **kwargs) -> Dict:
        """创建基础元数据"""
        path = Path(file_path)
        metadata = {
            "source": str(path.absolute()),
            "filename": path.name,
            "file_type": path.suffix.lower(),
            "file_size": os.path.getsize(file_path),
        }
        metadata.update(kwargs)
        return metadata


# ============================================================================
# 纯文本加载器 (.txt)
# ============================================================================
class TextLoader(BaseLoader):
    """纯文本文件加载器"""

    def __init__(self, encoding: str = "utf-8"):
        self.encoding = encoding

    def load(self, file_path: str) -> List[Document]:
        """加载 .txt 文件"""
        with open(file_path, "r", encoding=self.encoding) as f:
            content = f.read()

        metadata = self._create_metadata(file_path)
        return [Document(content=content, metadata=metadata)]


# ============================================================================
# Markdown 加载器 (.md)
# ============================================================================
class MarkdownLoader(BaseLoader):
    """Markdown 文件加载器，保留原始格式"""

    def __init__(self, encoding: str = "utf-8"):
        self.encoding = encoding

    def load(self, file_path: str) -> List[Document]:
        """加载 .md 文件"""
        with open(file_path, "r", encoding=self.encoding) as f:
            content = f.read()

        metadata = self._create_metadata(file_path, format="markdown")
        return [Document(content=content, metadata=metadata)]


# ============================================================================
# PDF 加载器 (.pdf)
# ============================================================================
class PDFLoader(BaseLoader):
    """
    PDF 文件加载器

    依赖 PyPDF2 库。每一页作为一个独立的 Document。
    """

    def load(self, file_path: str) -> List[Document]:
        """加载 .pdf 文件，按页拆分"""
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            raise ImportError(
                "加载 PDF 需要安装 PyPDF2: pip install PyPDF2"
            )

        reader = PdfReader(file_path)
        documents = []

        for page_num, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            if text and text.strip():  # 跳过空白页
                metadata = self._create_metadata(
                    file_path, page=page_num, total_pages=len(reader.pages)
                )
                documents.append(Document(content=text.strip(), metadata=metadata))

        return documents


# ============================================================================
# Word 文档加载器 (.docx)
# ============================================================================
class DocxLoader(BaseLoader):
    """
    Word 文档加载器

    依赖 python-docx 库。
    """

    def load(self, file_path: str) -> List[Document]:
        """加载 .docx 文件"""
        try:
            from docx import Document as DocxDocument
        except ImportError:
            raise ImportError(
                "加载 DOCX 需要安装 python-docx: pip install python-docx"
            )

        doc = DocxDocument(file_path)
        # 合并所有段落文本
        paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
        content = "\n".join(paragraphs)

        metadata = self._create_metadata(
            file_path, paragraph_count=len(paragraphs)
        )
        return [Document(content=content, metadata=metadata)]


# ============================================================================
# JSON 加载器 (.json)
# ============================================================================
class JSONLoader(BaseLoader):
    """
    JSON 文件加载器

    支持两种格式:
    1. 字符串数组: ["text1", "text2", ...]
    2. 对象数组: [{"content": "...", "title": "..."}, ...]
    """

    def __init__(self, content_key: str = "content", encoding: str = "utf-8"):
        self.content_key = content_key
        self.encoding = encoding

    def load(self, file_path: str) -> List[Document]:
        """加载 .json 文件"""
        with open(file_path, "r", encoding=self.encoding) as f:
            data = json.load(f)

        documents = []
        base_metadata = self._create_metadata(file_path)

        if isinstance(data, list):
            for idx, item in enumerate(data):
                if isinstance(item, str):
                    # 字符串数组格式
                    metadata = {**base_metadata, "index": idx}
                    documents.append(Document(content=item, metadata=metadata))
                elif isinstance(item, dict) and self.content_key in item:
                    # 对象数组格式
                    content = item[self.content_key]
                    extra_meta = {
                        k: v for k, v in item.items() if k != self.content_key
                    }
                    metadata = {**base_metadata, "index": idx, **extra_meta}
                    documents.append(Document(content=content, metadata=metadata))
        elif isinstance(data, dict) and self.content_key in data:
            # 单个对象格式
            documents.append(
                Document(content=data[self.content_key], metadata=base_metadata)
            )

        return documents


# ============================================================================
# 文档加载器工厂
# ============================================================================
class DocumentLoaderFactory:
    """
    文档加载器工厂（工厂模式）

    根据文件扩展名自动选择合适的加载器。
    支持注册自定义加载器以扩展支持的文件格式。

    使用示例:
        factory = DocumentLoaderFactory()
        docs = factory.load("path/to/file.pdf")

        # 注册自定义加载器
        factory.register(".csv", CSVLoader)
    """

    # 内置加载器映射
    _loaders: Dict[str, Type[BaseLoader]] = {
        ".txt": TextLoader,
        ".md": MarkdownLoader,
        ".pdf": PDFLoader,
        ".docx": DocxLoader,
        ".json": JSONLoader,
    }

    @classmethod
    def register(cls, extension: str, loader_class: Type[BaseLoader]):
        """
        注册新的加载器

        Args:
            extension: 文件扩展名（如 ".csv"）
            loader_class: 加载器类
        """
        cls._loaders[extension.lower()] = loader_class

    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """获取所有支持的文件扩展名"""
        return list(cls._loaders.keys())

    @classmethod
    def load(cls, file_path: str, **kwargs) -> List[Document]:
        """
        根据文件类型自动加载文档

        Args:
            file_path: 文件路径
            **kwargs: 传递给加载器的额外参数

        Returns:
            Document 对象列表

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 不支持的文件格式
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        ext = Path(file_path).suffix.lower()
        if ext not in cls._loaders:
            raise ValueError(
                f"不支持的文件格式: {ext}，"
                f"支持的格式: {cls.get_supported_extensions()}"
            )

        loader = cls._loaders[ext](**kwargs)
        return loader.load(file_path)

    @classmethod
    def load_directory(
        cls, directory: str, recursive: bool = True
    ) -> List[Document]:
        """
        加载目录下所有支持格式的文件

        Args:
            directory: 目录路径
            recursive: 是否递归加载子目录

        Returns:
            所有文档的列表
        """
        if not os.path.isdir(directory):
            raise NotADirectoryError(f"目录不存在: {directory}")

        documents = []
        supported = cls.get_supported_extensions()

        # 遍历目录
        for root, _, files in os.walk(directory):
            for filename in sorted(files):
                ext = Path(filename).suffix.lower()
                if ext in supported:
                    file_path = os.path.join(root, filename)
                    try:
                        docs = cls.load(file_path)
                        documents.extend(docs)
                        print(f"  [+] 已加载: {filename} ({len(docs)} 个文档片段)")
                    except Exception as e:
                        print(f"  [!] 加载失败: {filename} - {e}")

            if not recursive:
                break  # 不递归时只遍历顶层目录

        print(f"  [=] 共加载 {len(documents)} 个文档片段")
        return documents

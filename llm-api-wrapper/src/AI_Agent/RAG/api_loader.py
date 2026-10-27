"""
后台 API 数据加载模块

从 HTTP 后台接口拉取数据，并转换为统一的 Document 结构，
之后即可复用 RAG 的标准流程（分割 → 向量化 → 存储 → 检索）。

学习要点:
- 数据源不止本地文件，任何能返回文本/JSON 的后台接口都可作为知识来源
- 关键在于把接口返回的数据"抽取 + 映射"成统一的 Document
- 认证（token/cookie）、分页、字段映射是接入真实后台的三个常见问题

典型用法:
    from api_loader import APILoader
    from rag_pipeline import RAGPipeline

    loader = APILoader(headers={"Authorization": "Bearer xxx"})
    docs = loader.load(
        url="https://api.example.com/articles",
        records_path="data.items",        # JSON 中列表所在路径
        content_fields=["title", "body"],  # 用这些字段拼成正文
        metadata_fields=["id", "author"],  # 保留为元数据
    )

    pipeline = RAGPipeline()
    pipeline.add_documents(docs, save=True)   # 增量入库
    # 或 pipeline.build_knowledge_base(documents=docs)  # 重建知识库
"""

import json
from typing import Any, Callable, Dict, List, Optional

try:
    # httpx 已随 openai / mcp 一同安装，作为默认 HTTP 客户端
    import httpx
except ImportError:
    httpx = None

from .document_loader import Document


# ============================================================================
# API 数据加载器
# ============================================================================
class APILoader:
    """
    后台接口数据加载器

    负责: 发起 HTTP 请求 → 解析响应 → 抽取记录 → 映射为 Document 列表。
    支持 JSON 与纯文本两种响应，支持认证头、cookie、查询参数、请求体。
    """

    def __init__(
        self,
        headers: Optional[Dict[str, str]] = None,
        cookies: Optional[Dict[str, str]] = None,
        timeout: int = 30,
        verify_ssl: bool = True,
    ):
        """
        Args:
            headers: 请求头（放置鉴权 token 等），如 {"Authorization": "Bearer xxx"}
            cookies: cookie 字典（很多后台用会话 cookie 鉴权）
            timeout: 请求超时时间（秒）
            verify_ssl: 是否校验 SSL 证书（内网自签证书可设为 False）
        """
        if httpx is None:
            raise ImportError("需要安装 httpx: pip install httpx")

        self.headers = headers or {}
        self.cookies = cookies or {}
        self.timeout = timeout
        self.verify_ssl = verify_ssl

    # ------------------------------------------------------------------
    # 发起请求
    # ------------------------------------------------------------------
    def fetch(
        self,
        url: str,
        method: str = "GET",
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> httpx.Response:
        """
        发起 HTTP 请求并返回原始响应

        Args:
            url: 接口地址
            method: 请求方法 GET / POST 等
            params: URL 查询参数
            json_body: JSON 请求体（POST）
            data: 表单请求体（POST）

        Returns:
            httpx.Response 对象

        Raises:
            httpx.HTTPStatusError: 状态码非 2xx 时抛出
        """
        with httpx.Client(
            headers=self.headers,
            cookies=self.cookies,
            timeout=self.timeout,
            verify=self.verify_ssl,
        ) as client:
            response = client.request(
                method=method.upper(),
                url=url,
                params=params,
                json=json_body,
                data=data,
            )
            response.raise_for_status()  # 非 2xx 直接报错，便于排查
            return response

    # ------------------------------------------------------------------
    # 加载为 Document 列表
    # ------------------------------------------------------------------
    def load(
        self,
        url: str,
        *,
        method: str = "GET",
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        response_type: str = "json",
        records_path: Optional[str] = None,
        content_fields: Optional[List[str]] = None,
        metadata_fields: Optional[List[str]] = None,
        content_builder: Optional[Callable[[Any], str]] = None,
    ) -> List[Document]:
        """
        从接口加载数据并转换为 Document 列表

        Args:
            url: 接口地址
            method: 请求方法
            params: 查询参数
            json_body: JSON 请求体
            response_type: 响应类型 "json" | "text"
            records_path: JSON 中记录列表所在的点路径，如 "data.items"；
                          为 None 时会尝试自动识别（顶层是 list 直接用，
                          否则找第一个 list 类型的值）
            content_fields: 用哪些字段拼成正文；为 None 时序列化整条记录
            metadata_fields: 保留为元数据的字段
            content_builder: 自定义"记录 → 正文"的函数，优先级最高

        Returns:
            Document 列表
        """
        response = self.fetch(url, method=method, params=params, json_body=json_body)

        # 纯文本响应：整体作为一个 Document
        if response_type == "text":
            return [
                Document(
                    content=response.text,
                    metadata={"source": url, "source_type": "api_text"},
                )
            ]

        # JSON 响应：抽取记录列表并逐条映射
        payload = response.json()
        records = self._extract_records(payload, records_path)

        documents = []
        for idx, record in enumerate(records):
            content = self._build_content(record, content_fields, content_builder)
            if not content or not content.strip():
                continue  # 跳过空内容

            metadata = self._build_metadata(record, metadata_fields, url, idx)
            documents.append(Document(content=content, metadata=metadata))

        return documents

    def load_paginated(
        self,
        url: str,
        *,
        page_param: str = "page",
        size_param: str = "pageSize",
        start_page: int = 1,
        page_size: int = 50,
        max_pages: int = 20,
        **load_kwargs,
    ) -> List[Document]:
        """
        分页加载（适合数据量大的列表接口）

        逐页请求，直到某页返回空列表或达到 max_pages 上限。

        Args:
            url: 接口地址
            page_param: 页码参数名
            size_param: 每页大小参数名
            start_page: 起始页码
            page_size: 每页数量
            max_pages: 最大拉取页数（防止无限循环）
            **load_kwargs: 透传给 load() 的其他参数（records_path 等）

        Returns:
            所有页合并后的 Document 列表
        """
        all_docs: List[Document] = []
        base_params = dict(load_kwargs.pop("params", {}) or {})

        for page in range(start_page, start_page + max_pages):
            params = {**base_params, page_param: page, size_param: page_size}
            docs = self.load(url, params=params, **load_kwargs)
            if not docs:
                break  # 空页，认为已到末尾
            all_docs.extend(docs)

        return all_docs

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_records(payload: Any, records_path: Optional[str]) -> List[Any]:
        """从 JSON 响应中定位到记录列表"""
        # 指定了路径：按点路径逐级下钻，如 "data.items"
        if records_path:
            node = payload
            for key in records_path.split("."):
                if isinstance(node, dict):
                    node = node.get(key)
                else:
                    node = None
                    break
            if isinstance(node, list):
                return node
            if node is not None:
                return [node]
            return []

        # 未指定路径：自动识别
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            # 找第一个 list 类型的值（常见于 {"code":0,"data":[...]}）
            for value in payload.values():
                if isinstance(value, list):
                    return value
            # 没有 list，就把整个对象当作单条记录
            return [payload]
        return []

    @staticmethod
    def _build_content(
        record: Any,
        content_fields: Optional[List[str]],
        content_builder: Optional[Callable[[Any], str]],
    ) -> str:
        """把单条记录转换为正文文本"""
        # 1. 自定义构建函数优先
        if content_builder is not None:
            return content_builder(record)

        # 2. 记录本身就是字符串
        if isinstance(record, str):
            return record

        # 3. dict：按指定字段拼接
        if isinstance(record, dict):
            if content_fields:
                parts = []
                for field in content_fields:
                    value = record.get(field)
                    if value is not None:
                        parts.append(f"{field}: {value}")
                return "\n".join(parts)
            # 未指定字段：序列化整条记录
            return json.dumps(record, ensure_ascii=False, indent=2)

        # 4. 其他类型：直接字符串化
        return str(record)

    @staticmethod
    def _build_metadata(
        record: Any,
        metadata_fields: Optional[List[str]],
        url: str,
        idx: int,
    ) -> Dict[str, Any]:
        """构建 Document 的元数据"""
        metadata: Dict[str, Any] = {
            "source": url,
            "source_type": "api",
            "record_index": idx,
        }
        if isinstance(record, dict) and metadata_fields:
            for field in metadata_fields:
                if field in record:
                    value = record[field]
                    # 元数据只保留基本类型，避免向量库序列化问题
                    if isinstance(value, (str, int, float, bool)):
                        metadata[field] = value
        return metadata


# ============================================================================
# 便捷函数：一步从 API 灌入知识库
# ============================================================================
def ingest_from_api(
    pipeline,
    url: str,
    *,
    rebuild: bool = False,
    headers: Optional[Dict[str, str]] = None,
    cookies: Optional[Dict[str, str]] = None,
    **load_kwargs,
) -> int:
    """
    从后台 API 拉取数据并灌入 RAG 知识库

    Args:
        pipeline: RAGPipeline 实例
        url: 接口地址
        rebuild: True=清空并重建知识库；False=增量追加
        headers / cookies: 鉴权信息
        **load_kwargs: 透传给 APILoader.load()（records_path/content_fields 等）

    Returns:
        入库的文本块数量
    """
    loader = APILoader(headers=headers, cookies=cookies)
    docs = loader.load(url, **load_kwargs)
    print(f"[API] 从 {url} 获取到 {len(docs)} 条记录")

    if rebuild:
        return pipeline.build_knowledge_base(documents=docs)
    return pipeline.add_documents(docs, save=True)

"""
检索器模块

负责根据用户查询从向量库中检索最相关的文档。
包含多种进阶检索策略。

学习要点:
- 基础检索: 向量相似度搜索
- 进阶策略:
  1. 查询改写(Query Rewriting): 将用户问题改写为更适合检索的形式
  2. 多查询检索(Multi-Query): 生成多个角度的查询，合并结果
  3. 混合检索(Hybrid Search): 向量检索 + 关键词检索（BM25）
  4. 重排序(Reranking): 对初步结果用更精确的模型重新排序
  5. 最大边际相关性(MMR): 兼顾相关性和结果多样性
"""

from typing import Callable, List, Optional

from .config import RetrieverConfig
from .document_loader import Document
from .embeddings import BaseEmbedding
from .vector_store import BaseVectorStore, SearchResult


# ============================================================================
# 基础检索器
# ============================================================================
class Retriever:
    """
    文档检索器

    核心职责: 接收用户查询，返回最相关的文档列表。
    支持多种检索策略和进阶优化。
    """

    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedding_model: BaseEmbedding,
        config: Optional[RetrieverConfig] = None,
        llm_call: Optional[Callable] = None,
    ):
        """
        Args:
            vector_store: 向量存储实例
            embedding_model: 嵌入模型实例
            config: 检索器配置
            llm_call: LLM 调用函数（用于查询改写等需要 LLM 的功能）
        """
        if config is None:
            from .config import default_config
            config = default_config.retriever

        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.config = config
        self.llm_call = llm_call

    def retrieve(self, query: str) -> List[SearchResult]:
        """
        检索与查询最相关的文档

        完整检索流程:
        1. (可选) 查询改写
        2. 向量检索 / 混合检索
        3. 分数过滤
        4. (可选) MMR 多样性优化
        5. (可选) 重排序

        Args:
            query: 用户查询

        Returns:
            排序后的检索结果列表
        """
        # Step 1: 查询改写（进阶优化）
        search_query = query
        if self.config.enable_query_rewrite and self.llm_call:
            search_query = self._rewrite_query(query)
            print(f"  [查询改写] 原始: {query}")
            print(f"  [查询改写] 改写: {search_query}")

        # Step 2: 执行检索
        if self.config.search_type == "hybrid":
            results = self._hybrid_search(search_query)
        else:
            results = self._vector_search(search_query)

        # Step 3: 分数过滤
        results = self._filter_by_score(results)

        # Step 4: MMR 多样性优化
        if self.config.search_type == "mmr":
            results = self._mmr_rerank(search_query, results)

        # Step 5: 重排序（进阶优化）
        if self.config.enable_rerank and self.llm_call:
            results = self._rerank(query, results)

        return results[: self.config.top_k]

    # ========================================================================
    # 基础向量检索
    # ========================================================================
    def _vector_search(self, query: str) -> List[SearchResult]:
        """
        基础向量相似度检索

        流程: 查询文本 → 向量化 → 向量相似度搜索
        """
        query_embedding = self.embedding_model.embed_text(query)
        # 多取一些结果用于后续过滤
        fetch_k = self.config.top_k * 3
        results = self.vector_store.search(query_embedding, top_k=fetch_k)
        return results

    # ========================================================================
    # 混合检索（向量 + 关键词）
    # ========================================================================
    def _hybrid_search(self, query: str) -> List[SearchResult]:
        """
        混合检索策略

        结合向量语义检索和关键词匹配（BM25），
        综合两者的优势:
        - 向量检索: 捕获语义相似性（"猫" 和 "喵星人"）
        - 关键词检索: 精确匹配专有名词、代码等

        使用 RRF (Reciprocal Rank Fusion) 合并两种结果的排名。
        """
        # 向量检索结果
        vector_results = self._vector_search(query)

        # 关键词检索（简化版 BM25）
        keyword_results = self._keyword_search(query)

        # RRF 融合排名
        merged = self._reciprocal_rank_fusion(
            [vector_results, keyword_results], k=60
        )
        return merged

    def _keyword_search(self, query: str) -> List[SearchResult]:
        """
        简化的关键词检索

        基于词频匹配（TF）进行简单的关键词搜索。
        生产环境建议使用 Elasticsearch 或专业的 BM25 实现。
        """
        import re

        # 简单分词（按非中文字符和空格分割）
        query_terms = set(re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+', query.lower()))

        results = []
        for idx, doc in enumerate(self.vector_store.documents):
            content_lower = doc.content.lower()
            # 计算匹配的关键词比例
            matched = sum(1 for term in query_terms if term in content_lower)
            if matched > 0:
                score = matched / len(query_terms)
                results.append(SearchResult(document=doc, score=score, rank=idx))

        # 按分数排序
        results.sort(key=lambda x: x.score, reverse=True)
        return results[: self.config.top_k * 3]

    def _reciprocal_rank_fusion(
        self, result_lists: List[List[SearchResult]], k: int = 60
    ) -> List[SearchResult]:
        """
        RRF (Reciprocal Rank Fusion) 排名融合

        将多个排名列表合并为一个统一排名。
        公式: score(d) = Σ 1 / (k + rank_i(d))
        k 值越大，排名的影响越平滑。
        """
        doc_scores = {}  # content -> (score, document)

        for results in result_lists:
            for rank, result in enumerate(results):
                key = result.document.content[:100]  # 用内容前100字符作为去重键
                rrf_score = 1.0 / (k + rank + 1)
                if key in doc_scores:
                    doc_scores[key] = (
                        doc_scores[key][0] + rrf_score,
                        result.document,
                    )
                else:
                    doc_scores[key] = (rrf_score, result.document)

        # 按融合分数排序
        sorted_results = sorted(doc_scores.values(), key=lambda x: x[0], reverse=True)
        return [
            SearchResult(document=doc, score=score, rank=rank)
            for rank, (score, doc) in enumerate(sorted_results)
        ]

    # ========================================================================
    # 分数过滤
    # ========================================================================
    def _filter_by_score(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        按相似度阈值过滤结果

        移除相似度得分低于阈值的结果，减少噪声。
        """
        if self.config.score_threshold <= 0:
            return results
        return [r for r in results if r.score >= self.config.score_threshold]

    # ========================================================================
    # MMR 多样性优化
    # ========================================================================
    def _mmr_rerank(
        self, query: str, results: List[SearchResult]
    ) -> List[SearchResult]:
        """
        最大边际相关性 (Maximal Marginal Relevance)

        在保持相关性的同时增加结果的多样性，
        避免返回大量内容重复的文档。

        MMR = λ * sim(q, d) - (1-λ) * max(sim(d, d_selected))

        其中:
        - λ 越大越偏向相关性
        - λ 越小越偏向多样性
        """
        if len(results) <= 1:
            return results

        import numpy as np

        # 获取所有结果的向量
        texts = [r.document.content for r in results]
        embeddings = self.embedding_model.embed_texts(texts)
        query_embedding = self.embedding_model.embed_text(query)

        # 转换为 numpy 数组
        doc_vectors = np.array(embeddings)
        query_vector = np.array(query_embedding)

        # MMR 选择
        selected_indices = []
        remaining_indices = list(range(len(results)))
        lambda_param = self.config.mmr_lambda

        while len(selected_indices) < min(self.config.top_k, len(results)):
            best_score = -float("inf")
            best_idx = -1

            for idx in remaining_indices:
                # 与查询的相似度
                relevance = np.dot(doc_vectors[idx], query_vector)

                # 与已选文档的最大相似度
                if selected_indices:
                    selected_vectors = doc_vectors[selected_indices]
                    redundancy = max(
                        np.dot(doc_vectors[idx], sv)
                        for sv in selected_vectors
                    )
                else:
                    redundancy = 0

                # MMR 分数
                mmr_score = lambda_param * relevance - (1 - lambda_param) * redundancy
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            if best_idx >= 0:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)

        return [
            SearchResult(
                document=results[idx].document,
                score=results[idx].score,
                rank=rank,
            )
            for rank, idx in enumerate(selected_indices)
        ]

    # ========================================================================
    # 进阶: 查询改写
    # ========================================================================
    def _rewrite_query(self, query: str) -> str:
        """
        查询改写

        将用户的口语化查询改写为更适合检索的形式。
        例如: "RAG怎么用" → "RAG 检索增强生成的使用方法和实现步骤"
        """
        rewrite_prompt = f"""请将以下用户问题改写为更适合从知识库中检索的查询语句。
要求:
1. 保持原始语义不变
2. 补充关键词和同义词
3. 使用更正式、更具体的表述
4. 只输出改写后的查询，不要其他内容

用户问题: {query}
改写后的查询:"""

        try:
            rewritten = self.llm_call(rewrite_prompt)
            return rewritten.strip() if rewritten else query
        except Exception:
            return query

    # ========================================================================
    # 进阶: 重排序
    # ========================================================================
    def _rerank(
        self, query: str, results: List[SearchResult]
    ) -> List[SearchResult]:
        """
        LLM 重排序

        使用 LLM 对初步检索结果进行精细重排序。
        原理: 让 LLM 判断每个文档与查询的相关程度，
        比纯向量相似度更准确（但更慢、更贵）。

        注意: 生产环境建议使用专门的 Rerank 模型（如 Cohere Rerank、BGE-Reranker）
        """
        if not results:
            return results

        rerank_prompt = f"""请根据用户问题，对以下文档片段按相关性从高到低排序。
只输出排序后的编号列表（如: 2,1,3,4），不要其他内容。

用户问题: {query}

文档片段:
"""
        for i, result in enumerate(results[:10]):  # 最多对前10个重排
            content_preview = result.document.content[:200]
            rerank_prompt += f"\n[{i+1}] {content_preview}\n"

        rerank_prompt += "\n排序结果（编号用逗号分隔）:"

        try:
            response = self.llm_call(rerank_prompt)
            # 解析排序结果
            import re
            numbers = re.findall(r'\d+', response)
            order = [int(n) - 1 for n in numbers if 0 < int(n) <= len(results)]

            # 按新顺序重排
            reranked = []
            seen = set()
            for idx in order:
                if idx not in seen and idx < len(results):
                    seen.add(idx)
                    reranked.append(
                        SearchResult(
                            document=results[idx].document,
                            score=results[idx].score,
                            rank=len(reranked),
                        )
                    )
            # 添加未被排序到的文档
            for idx, result in enumerate(results):
                if idx not in seen:
                    reranked.append(
                        SearchResult(
                            document=result.document,
                            score=result.score,
                            rank=len(reranked),
                        )
                    )
            return reranked

        except Exception:
            # 重排失败时保持原序
            return results

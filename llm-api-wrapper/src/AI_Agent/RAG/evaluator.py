# ============================================================================
# 【补充模块 · RAG 评估系统核心】
# 新增时间: 2026-07-23
# 说明: 本文件为后续补充，独立于最初的 RAG demo 核心模块(config/retriever/...)。
#       用于量化评测 RAG 系统的回答质量，是「评估驱动开发」的落地工具。
# ----------------------------------------------------------------------------
# 本模块实现一套精简版 RAGAS 风格评估，包含三类指标：
#   A. LLM-as-judge（用 LLM 当裁判，需要调用 API）
#      - faithfulness        忠实度：答案是否忠于检索到的上下文（不编造）
#      - answer_relevancy    答案相关性：答案是否切题、完整
#      - answer_correctness  答案正确性：答案与参考答案是否一致
#      - context_relevance   上下文相关性：检索到的上下文是否与问题相关
#   B. 检索指标（无需 LLM，靠期望来源判断）
#      - hit_rate            命中率：期望文档是否被检索到
#      - mrr                 平均倒数排名：期望文档排在第几位
#   C. 词面指标（无需 LLM，便宜的离线基线）
#      - lexical_f1          答案与参考答案的字符二元组 F1 重合度
# ============================================================================

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .config import LLMConfig, default_config


# ============================================================================
# 数据结构
# ============================================================================
@dataclass
class EvalSample:
    """单条评测样本"""

    question: str
    ground_truth: str = ""                       # 参考答案（人工标注）
    expected_sources: List[str] = field(default_factory=list)  # 期望命中的文档名


@dataclass
class SampleResult:
    """单条样本的评测结果"""

    question: str
    answer: str
    retrieved_sources: List[str]
    scores: Dict[str, float]        # 各项指标得分
    elapsed_time: float = 0.0


@dataclass
class EvalReport:
    """整体评测报告"""

    sample_results: List[SampleResult]
    aggregate: Dict[str, float]     # 各指标的平均分
    avg_latency: float = 0.0


# ============================================================================
# 评估器
# ============================================================================
class RAGEvaluator:
    """
    RAG 评估器

    用法:
        evaluator = RAGEvaluator()
        report = evaluator.evaluate(pipeline, samples)
        print(evaluator.format_report(report))
    """

    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        use_llm_judge: bool = True,
    ):
        """
        Args:
            llm_config: 裁判 LLM 配置，默认复用全局配置
            use_llm_judge: 是否启用 LLM 裁判指标（关掉可省钱，只跑检索+词面指标）
        """
        self.llm_config = llm_config or default_config.llm
        self.use_llm_judge = use_llm_judge
        if use_llm_judge:
            self._init_judge_client()

    def _init_judge_client(self):
        """初始化裁判用的 LLM 客户端（独立于生成器，避免互相影响）"""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("需要安装 openai: pip install openai")
        self.client = OpenAI(
            api_key=self.llm_config.api_key,
            base_url=self.llm_config.base_url,
            timeout=self.llm_config.timeout,
        )

    # ========================================================================
    # 主流程
    # ========================================================================
    def evaluate(self, pipeline, samples: List[EvalSample]) -> EvalReport:
        """
        在评测集上评估一个 RAG 管道

        Args:
            pipeline: RAGPipeline 实例
            samples: 评测样本列表

        Returns:
            EvalReport 评测报告
        """
        # 评测期间临时关闭管道的调试打印，保持输出整洁
        original_debug = pipeline.debug
        pipeline.debug = False
        pipeline.generator.clear_history()  # 评测每条独立，不带对话历史

        sample_results: List[SampleResult] = []
        total = len(samples)

        for i, sample in enumerate(samples, 1):
            print(f"  [{i}/{total}] 评测中: {sample.question[:30]}...")

            # 1. 跑一次完整 RAG 查询
            response = pipeline.query(sample.question, use_citation=False)
            contexts = [s.document.content for s in response.sources]
            retrieved_sources = [
                s.document.metadata.get("filename", "未知") for s in response.sources
            ]

            # 2. 计算各项指标
            scores: Dict[str, float] = {}

            # --- B. 检索指标 ---
            if sample.expected_sources:
                scores["hit_rate"] = self._retrieval_hit(
                    sample.expected_sources, retrieved_sources
                )
                scores["mrr"] = self._retrieval_mrr(
                    sample.expected_sources, retrieved_sources
                )

            # --- C. 词面指标 ---
            if sample.ground_truth:
                scores["lexical_f1"] = self._lexical_f1(
                    response.answer, sample.ground_truth
                )

            # --- A. LLM 裁判指标 ---
            if self.use_llm_judge:
                scores["faithfulness"] = self._judge_faithfulness(
                    response.answer, contexts
                )
                scores["answer_relevancy"] = self._judge_answer_relevancy(
                    sample.question, response.answer
                )
                scores["context_relevance"] = self._judge_context_relevance(
                    sample.question, contexts
                )
                if sample.ground_truth:
                    scores["answer_correctness"] = self._judge_answer_correctness(
                        sample.question, response.answer, sample.ground_truth
                    )

            sample_results.append(
                SampleResult(
                    question=sample.question,
                    answer=response.answer,
                    retrieved_sources=retrieved_sources,
                    scores=scores,
                    elapsed_time=response.elapsed_time,
                )
            )

        pipeline.debug = original_debug  # 恢复原始调试开关

        # 3. 汇总
        aggregate = self._aggregate(sample_results)
        avg_latency = (
            sum(r.elapsed_time for r in sample_results) / len(sample_results)
            if sample_results
            else 0.0
        )
        return EvalReport(
            sample_results=sample_results,
            aggregate=aggregate,
            avg_latency=avg_latency,
        )

    # ========================================================================
    # A. LLM 裁判指标
    # ========================================================================
    def _judge_faithfulness(self, answer: str, contexts: List[str]) -> float:
        """忠实度：答案中的信息是否都能在上下文中找到依据（惩罚幻觉）"""
        context_text = "\n\n".join(contexts) if contexts else "（无上下文）"
        prompt = f"""你是严格的评估专家。请判断「回答」中的信息是否都能从「参考上下文」中得到支持。
评分标准(0到1)：
- 1.0 = 回答的所有关键信息都有上下文支持，没有编造
- 0.5 = 部分信息有支持，部分无法从上下文得到
- 0.0 = 回答大量信息无上下文依据或与上下文矛盾

参考上下文:
{context_text}

回答:
{answer}

只输出 JSON，格式: {{"score": 分数, "reason": "简短理由"}}"""
        return self._judge(prompt)

    def _judge_answer_relevancy(self, question: str, answer: str) -> float:
        """答案相关性：答案是否切题、是否直接回应了问题"""
        prompt = f"""你是严格的评估专家。请判断「回答」与「问题」的相关程度。
评分标准(0到1)：
- 1.0 = 直接、完整地回应了问题
- 0.5 = 部分相关或答非所问一部分
- 0.0 = 完全跑题或没有回答问题

问题: {question}

回答: {answer}

只输出 JSON，格式: {{"score": 分数, "reason": "简短理由"}}"""
        return self._judge(prompt)

    def _judge_answer_correctness(
        self, question: str, answer: str, ground_truth: str
    ) -> float:
        """答案正确性：答案与参考答案在事实上是否一致"""
        prompt = f"""你是严格的评估专家。请对比「模型回答」与「参考答案」，判断事实是否一致。
评分标准(0到1)：
- 1.0 = 关键事实与参考答案完全一致
- 0.5 = 部分事实正确，有遗漏或小错误
- 0.0 = 事实错误或与参考答案矛盾
注意：表述方式可以不同，只看事实是否正确。

问题: {question}

参考答案: {ground_truth}

模型回答: {answer}

只输出 JSON，格式: {{"score": 分数, "reason": "简短理由"}}"""
        return self._judge(prompt)

    def _judge_context_relevance(self, question: str, contexts: List[str]) -> float:
        """上下文相关性：检索到的上下文整体与问题的相关程度"""
        if not contexts:
            return 0.0
        context_text = "\n\n".join(
            f"[片段{i+1}] {c[:300]}" for i, c in enumerate(contexts)
        )
        prompt = f"""你是严格的评估专家。请判断检索到的「上下文片段」整体与「问题」的相关程度。
评分标准(0到1)：
- 1.0 = 片段高度相关，足以回答问题
- 0.5 = 部分片段相关，混有无关内容
- 0.0 = 片段基本与问题无关

问题: {question}

上下文片段:
{context_text}

只输出 JSON，格式: {{"score": 分数, "reason": "简短理由"}}"""
        return self._judge(prompt)

    def _judge(self, prompt: str) -> float:
        """调用裁判 LLM 并解析出分数（低温度保证评分稳定）"""
        try:
            response = self.client.chat.completions.create(
                model=self.llm_config.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,  # 评分要确定性，温度设 0
                max_tokens=300,
            )
            text = response.choices[0].message.content.strip()
            data = self._parse_json(text)
            score = float(data.get("score", 0.0))
            # 规约到 [0,1]
            return max(0.0, min(1.0, score))
        except Exception as e:
            print(f"    [!] 裁判评分失败: {e}")
            return 0.0

    @staticmethod
    def _parse_json(text: str) -> dict:
        """从 LLM 输出中稳健地提取 JSON（容忍 markdown 代码块包裹）"""
        text = text.strip()
        # 去掉 ```json ... ``` 包裹
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE)
        # 提取第一个 { 到最后一个 }
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        return {}

    # ========================================================================
    # B. 检索指标（无需 LLM）
    # ========================================================================
    @staticmethod
    def _retrieval_hit(expected: List[str], retrieved: List[str]) -> float:
        """命中率：期望来源中有任意一个被检索到即算命中(1.0)"""
        retrieved_set = set(retrieved)
        return 1.0 if any(e in retrieved_set for e in expected) else 0.0

    @staticmethod
    def _retrieval_mrr(expected: List[str], retrieved: List[str]) -> float:
        """
        平均倒数排名 (Mean Reciprocal Rank)
        第一个期望来源出现的排名越靠前，得分越高(1/rank)
        """
        for rank, source in enumerate(retrieved, 1):
            if source in expected:
                return 1.0 / rank
        return 0.0

    # ========================================================================
    # C. 词面指标（无需 LLM）
    # ========================================================================
    @staticmethod
    def _lexical_f1(answer: str, ground_truth: str) -> float:
        """
        基于字符二元组(bigram)的 F1 相似度

        这是一个便宜的离线基线指标：衡量答案与参考答案的字面重合度。
        不理解语义，仅作为 LLM 裁判之外的廉价参考。
        """
        def bigrams(text: str) -> set:
            text = re.sub(r"\s+", "", text)
            return {text[i : i + 2] for i in range(len(text) - 1)}

        a, b = bigrams(answer), bigrams(ground_truth)
        if not a or not b:
            return 0.0
        overlap = len(a & b)
        if overlap == 0:
            return 0.0
        precision = overlap / len(a)
        recall = overlap / len(b)
        return 2 * precision * recall / (precision + recall)

    # ========================================================================
    # 汇总与报告
    # ========================================================================
    @staticmethod
    def _aggregate(results: List[SampleResult]) -> Dict[str, float]:
        """计算每项指标的平均分"""
        if not results:
            return {}
        all_metrics: Dict[str, List[float]] = {}
        for r in results:
            for metric, score in r.scores.items():
                all_metrics.setdefault(metric, []).append(score)
        return {m: sum(v) / len(v) for m, v in all_metrics.items()}

    # 指标中文名映射（报告展示用）
    METRIC_NAMES = {
        "faithfulness": "忠实度(不幻觉)",
        "answer_relevancy": "答案相关性",
        "answer_correctness": "答案正确性",
        "context_relevance": "上下文相关性",
        "hit_rate": "检索命中率",
        "mrr": "检索MRR",
        "lexical_f1": "词面F1",
    }

    def format_report(self, report: EvalReport) -> str:
        """把评测报告格式化为可读文本"""
        lines = []
        lines.append("=" * 64)
        lines.append("RAG 评估报告")
        lines.append("=" * 64)

        # 汇总分数
        lines.append("\n【整体指标（平均分，满分 1.0）】")
        for metric, score in report.aggregate.items():
            name = self.METRIC_NAMES.get(metric, metric)
            bar = self._score_bar(score)
            lines.append(f"  {name:<16} {score:.3f}  {bar}")
        lines.append(f"\n  平均响应延迟: {report.avg_latency:.2f}s")
        lines.append(f"  样本数量: {len(report.sample_results)}")

        # 逐条明细
        lines.append("\n" + "─" * 64)
        lines.append("【逐条明细】")
        for i, r in enumerate(report.sample_results, 1):
            lines.append(f"\n[{i}] 问题: {r.question}")
            score_str = "  ".join(
                f"{self.METRIC_NAMES.get(m, m)}={s:.2f}" for m, s in r.scores.items()
            )
            lines.append(f"    得分: {score_str}")
            lines.append(f"    命中来源: {r.retrieved_sources}")
            lines.append(f"    答案: {r.answer[:100]}...")

        lines.append("\n" + "=" * 64)
        return "\n".join(lines)

    @staticmethod
    def _score_bar(score: float, width: int = 20) -> str:
        """用字符条直观展示分数"""
        filled = int(round(score * width))
        return "█" * filled + "░" * (width - filled)


# ============================================================================
# 评测集加载
# ============================================================================
def load_eval_dataset(path: str) -> List[EvalSample]:
    """从 JSON 文件加载评测集"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = []
    for item in data.get("samples", []):
        samples.append(
            EvalSample(
                question=item["question"],
                ground_truth=item.get("ground_truth", ""),
                expected_sources=item.get("expected_sources", []),
            )
        )
    return samples

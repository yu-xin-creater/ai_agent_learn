# ============================================================================
# 【补充脚本 · RAG 评估跑分入口】
# 新增时间: 2026-07-23
# 说明: 此脚本为后续补充。一键完成：加载评测集 → 构建/加载知识库 →
#       在评测集上跑 RAG → 用 evaluator 打分 → 打印并保存评估报告。
# ----------------------------------------------------------------------------
# 运行方式:
#   python src/AI_Agent/RAG/run_eval.py                 # 完整评估(含LLM裁判)
#   python src/AI_Agent/RAG/run_eval.py --no-llm-judge  # 只跑检索+词面指标(省钱)
#   python -m src.AI_Agent.RAG.run_eval                 # 模块方式
# ============================================================================

import argparse
import datetime
import json
import os
import sys

# ---- 导入兼容处理：支持脚本直接运行与模块运行（与 main.py 一致）----
if __package__ in (None, ""):
    _PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PARENT_DIR not in sys.path:
        sys.path.insert(0, _PARENT_DIR)
    import RAG  # noqa: F401
    __package__ = "RAG"

from .config import default_config
from .evaluator import RAGEvaluator, load_eval_dataset
from .rag_pipeline import RAGPipeline


# 补充时间常量（写入报告，方便区分这是后加的评估产物）
SUPPLEMENT_DATE = "2026-07-23"


def main():
    parser = argparse.ArgumentParser(description="RAG 评估跑分 (补充于 2026-07-23)")
    parser.add_argument(
        "--no-llm-judge",
        action="store_true",
        help="关闭 LLM 裁判指标，仅跑检索命中/MRR/词面F1（不消耗生成费用）",
    )
    parser.add_argument(
        "--dataset",
        default=os.path.join(os.path.dirname(__file__), "eval_dataset.json"),
        help="评测集 JSON 路径",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="不保存报告文件，仅打印"
    )
    args = parser.parse_args()

    print("=" * 64)
    print(f"RAG 评估系统 (补充于 {SUPPLEMENT_DATE})")
    print("=" * 64)

    # ---- 1. 检查 API Key ----
    config = default_config
    placeholder_keys = {"", "your_api_key_here", "your_api_key"}
    if config.llm.api_key in placeholder_keys:
        print("[!] 未配置有效的 API Key，请先在 .env 中设置 LLM_API_KEY")
        sys.exit(1)

    # ---- 2. 加载评测集 ----
    print(f"\n[1/4] 加载评测集: {args.dataset}")
    samples = load_eval_dataset(args.dataset)
    print(f"      共 {len(samples)} 条评测样本")

    # ---- 3. 构建/加载知识库 ----
    print("\n[2/4] 准备知识库...")
    pipeline = RAGPipeline(config=config)
    vector_index = os.path.join(config.vector_store.persist_directory, "faiss.index")
    if os.path.exists(vector_index):
        try:
            pipeline.load_knowledge_base()
        except Exception:
            pipeline.build_knowledge_base(config.knowledge_base_dir)
    else:
        pipeline.build_knowledge_base(config.knowledge_base_dir)

    # ---- 4. 运行评估 ----
    use_llm_judge = not args.no_llm_judge
    print(f"\n[3/4] 开始评估 (LLM裁判: {'开启' if use_llm_judge else '关闭'})...")
    evaluator = RAGEvaluator(use_llm_judge=use_llm_judge)
    report = evaluator.evaluate(pipeline, samples)

    # ---- 5. 输出报告 ----
    print("\n[4/4] 评估完成\n")
    report_text = evaluator.format_report(report)
    print(report_text)

    # 保存报告（文件名带时间戳，方便区分每次跑分）
    if not args.no_save:
        _save_report(report, report_text)


def _save_report(report, report_text: str):
    """保存 Markdown + JSON 两种格式的报告，文件名含时间戳"""
    report_dir = os.path.join(os.path.dirname(__file__), "eval_reports")
    os.makedirs(report_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Markdown 报告（给人看）
    md_path = os.path.join(report_dir, f"eval_report_{timestamp}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# RAG 评估报告\n\n")
        f.write(f"> 评估系统补充于 2026-07-23 | 本次跑分时间: {timestamp}\n\n")
        f.write("```\n")
        f.write(report_text)
        f.write("\n```\n")

    # JSON 报告（给程序/做趋势对比用）
    json_path = os.path.join(report_dir, f"eval_report_{timestamp}.json")
    json_data = {
        "supplement_date": "2026-07-23",
        "run_time": timestamp,
        "aggregate": report.aggregate,
        "avg_latency": report.avg_latency,
        "samples": [
            {
                "question": r.question,
                "answer": r.answer,
                "retrieved_sources": r.retrieved_sources,
                "scores": r.scores,
                "elapsed_time": r.elapsed_time,
            }
            for r in report.sample_results
        ],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    print(f"\n[报告已保存]")
    print(f"  Markdown: {md_path}")
    print(f"  JSON:     {json_path}")


if __name__ == "__main__":
    main()

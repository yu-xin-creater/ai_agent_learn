"""
RAG 知识问答系统 - 主入口

提供交互式命令行界面，演示 RAG 的完整工作流。

使用方法:
    1. 确保 .env 文件已配置 LLM_API_KEY
    2. 运行: python -m src.AI_Agent.RAG.main
    或者: python src/AI_Agent/RAG/main.py

功能:
    - 自动加载 knowledge_base 目录下的文档
    - 交互式问答
    - 支持多轮对话
    - 支持动态添加文档
"""

import os
import sys

# ============================================================================
# 导入兼容处理
# ----------------------------------------------------------------------------
# 支持两种运行方式:
#   1. 模块运行: python -m src.AI_Agent.RAG.main  (推荐)
#   2. 脚本运行: python src/AI_Agent/RAG/main.py
#
# 直接以脚本方式运行时，Python 会把 main.py 当作顶层模块(__main__)，
# 此时不存在父包，相对导入(from .config)会报错。
# 解决办法：把 RAG 的父目录加入 sys.path，将 RAG 注册为可导入的顶层包，
# 并显式设置 __package__，使模块内的相对导入能够正确解析。
# ============================================================================
if __package__ in (None, ""):
    _PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PARENT_DIR not in sys.path:
        sys.path.insert(0, _PARENT_DIR)
    import RAG  # noqa: F401  确保 RAG 包被加载到 sys.modules
    __package__ = "RAG"


def print_banner():
    """打印欢迎横幅"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║            RAG 知识问答系统 Demo v1.0                        ║
║                                                              ║
║  基于检索增强生成（Retrieval-Augmented Generation）          ║
║  支持: 文档加载 → 分割 → 向量化 → 检索 → 生成              ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def print_help():
    """打印帮助信息"""
    help_text = """
可用命令:
  /help     - 显示此帮助信息
  /status   - 查看知识库状态
  /add      - 动态添加文本到知识库
  /clear    - 清空对话历史
  /stream   - 流式问答模式（输入问题后流式输出）
  /rebuild  - 重新构建知识库
  /quit     - 退出程序

直接输入问题即可开始问答。
"""
    print(help_text)


def main():
    """主函数"""
    from .config import RAGConfig, default_config
    from .rag_pipeline import RAGPipeline

    print_banner()

    # ========================================================================
    # 初始化配置
    # ========================================================================
    config = default_config

    # 检查 API Key（同时识别未替换的占位符）
    placeholder_keys = {"", "your_api_key_here", "your_api_key"}
    if config.llm.api_key in placeholder_keys:
        print("[!] 错误: 尚未配置有效的 LLM_API_KEY")
        print(f"    请编辑 .env 文件: {os.path.join(os.path.dirname(__file__), '.env')}")
        print("    将 LLM_API_KEY 和 EMBEDDING_API_KEY 替换为你的真实 API Key")
        print("    (通义千问密钥申请: https://bailian.console.aliyun.com/)")
        sys.exit(1)

    # ========================================================================
    # 初始化 RAG Pipeline
    # ========================================================================
    print("[系统] 正在初始化 RAG Pipeline...\n")
    pipeline = RAGPipeline(config=config)

    # ========================================================================
    # 构建知识库
    # ========================================================================
    knowledge_dir = config.knowledge_base_dir
    vector_db_dir = config.vector_store.persist_directory

    # 检查是否已有向量库
    if os.path.exists(os.path.join(vector_db_dir, "faiss.index")):
        print("[系统] 发现已有向量库，正在加载...")
        try:
            pipeline.load_knowledge_base()
        except Exception as e:
            print(f"[!] 加载失败: {e}，将重新构建")
            pipeline.build_knowledge_base(knowledge_dir)
    else:
        # 首次运行，构建知识库
        if os.path.exists(knowledge_dir) and os.listdir(knowledge_dir):
            pipeline.build_knowledge_base(knowledge_dir)
        else:
            print(f"[!] 知识库目录为空: {knowledge_dir}")
            print("    请将文档放入该目录，或使用 /add 命令动态添加文本")
            print("    支持格式: .txt, .md, .pdf, .docx, .json\n")

    # ========================================================================
    # 交互式问答循环
    # ========================================================================
    print_help()
    print("─" * 60)
    print("准备就绪! 请输入问题 (输入 /quit 退出):\n")

    while True:
        try:
            user_input = input("🙋 问题: ").strip()

            if not user_input:
                continue

            # 处理命令
            if user_input.startswith("/"):
                handle_command(user_input, pipeline, config)
                continue

            # 正常问答
            print("\n🤖 正在思考...\n")
            response = pipeline.query(user_input)
            print(response.format_answer())

        except KeyboardInterrupt:
            print("\n\n[系统] 收到中断信号，正在退出...")
            break
        except Exception as e:
            print(f"\n[!] 错误: {e}\n")


def handle_command(command: str, pipeline, config):
    """处理用户命令"""
    cmd = command.lower().split()[0]

    if cmd == "/help":
        print_help()

    elif cmd == "/status":
        print(f"\n[状态]")
        print(f"  知识库文档块数: {pipeline.knowledge_base_size}")
        print(f"  对话轮数: {len(pipeline.generator.chat_history) // 2}")
        print(f"  LLM 模型: {config.llm.model_name}")
        print(f"  嵌入模型: {config.embedding.model_name}")
        print(f"  检索策略: {config.retriever.search_type}")
        print(f"  向量库类型: {config.vector_store.store_type}")
        print()

    elif cmd == "/add":
        print("请输入要添加到知识库的文本（输入空行结束）:")
        lines = []
        while True:
            line = input()
            if line == "":
                break
            lines.append(line)
        if lines:
            text = "\n".join(lines)
            pipeline.add_document(text, metadata={"source": "user_input"})
            print(f"[+] 已添加 {len(text)} 字符的文本到知识库\n")
        else:
            print("[!] 未输入任何文本\n")

    elif cmd == "/clear":
        pipeline.clear_history()
        print("[+] 对话历史已清空\n")

    elif cmd == "/stream":
        question = input("🙋 请输入问题 (流式输出): ").strip()
        if question:
            print("\n🤖 ", end="", flush=True)
            for chunk in pipeline.query_stream(question):
                print(chunk, end="", flush=True)
            print("\n")

    elif cmd == "/rebuild":
        print("[系统] 重新构建知识库...")
        pipeline.build_knowledge_base(config.knowledge_base_dir)

    elif cmd == "/quit":
        print("[系统] 再见!")
        sys.exit(0)

    else:
        print(f"[!] 未知命令: {command}，输入 /help 查看帮助\n")


# ============================================================================
# 入口
# ============================================================================
if __name__ == "__main__":
    main()

"""
生成模块

负责调用 LLM 生成最终答案。
核心工作: 将检索到的上下文与用户问题组合成 Prompt，
然后调用 LLM 生成有据可循的回答。

学习要点:
- Prompt 工程: 如何构建有效的 RAG Prompt
- 上下文注入: 将检索结果嵌入 Prompt
- 进阶优化:
  - 上下文压缩: 只保留与问题相关的部分
  - 引用溯源: 生成回答时标注信息来源
  - 流式输出: 实时返回生成结果
  - 对话记忆: 支持多轮对话
"""

from typing import Generator, List, Optional

from .config import LLMConfig
from .vector_store import SearchResult


# ============================================================================
# Prompt 模板
# ============================================================================
# 基础 RAG Prompt 模板
RAG_PROMPT_TEMPLATE = """你是一个专业的知识问答助手。请根据以下提供的参考资料回答用户的问题。

要求:
1. 只根据参考资料中的信息回答，不要编造或推测
2. 如果参考资料中没有相关信息，请明确说明"根据现有资料无法回答该问题"
3. 回答要简洁准确，条理清晰
4. 在回答末尾标注信息来源

参考资料:
{context}

用户问题: {question}

请回答:"""

# 带引用的 Prompt 模板（进阶：回答时标注来源）
RAG_PROMPT_WITH_CITATION = """你是一个专业的知识问答助手。请根据以下编号的参考资料回答用户的问题。

要求:
1. 只根据参考资料中的信息回答，不要编造或推测
2. 回答中引用信息时，使用 [来源X] 的格式标注出处
3. 如果参考资料中没有相关信息，请明确说明
4. 回答要简洁准确，条理清晰

参考资料:
{context}

用户问题: {question}

请回答（注意标注来源）:"""

# 上下文压缩 Prompt（进阶：筛选相关内容）
CONTEXT_COMPRESSION_PROMPT = """请从以下文本中提取与问题相关的关键信息。
只保留直接回答问题所需的内容，去除无关部分。
如果文本与问题完全无关，输出"无关"。

问题: {question}
文本: {text}

相关信息:"""

# 多轮对话 Prompt 模板
CONVERSATIONAL_RAG_PROMPT = """你是一个专业的知识问答助手。根据参考资料和对话历史回答问题。

参考资料:
{context}

对话历史:
{chat_history}

用户问题: {question}

请回答:"""


# ============================================================================
# 生成器类
# ============================================================================
class RAGGenerator:
    """
    RAG 生成器

    负责将检索结果和用户问题组合成 Prompt，
    调用 LLM 生成最终答案。
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Args:
            config: LLM 配置
        """
        if config is None:
            from .config import default_config
            config = default_config.llm

        self.config = config
        self._init_client()

        # 对话历史（支持多轮对话）
        self.chat_history: List[dict] = []

    def _init_client(self):
        """初始化 OpenAI 兼容客户端"""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("需要安装 openai: pip install openai")

        self.client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout,
        )

    # ========================================================================
    # 核心生成方法
    # ========================================================================
    def generate(
        self,
        question: str,
        search_results: List[SearchResult],
        use_citation: bool = True,
        compress_context: bool = False,
    ) -> str:
        """
        生成 RAG 回答

        Args:
            question: 用户问题
            search_results: 检索结果列表
            use_citation: 是否使用引用格式
            compress_context: 是否压缩上下文

        Returns:
            LLM 生成的答案
        """
        # Step 1: 构建上下文
        context = self._build_context(
            question, search_results, compress_context
        )

        # Step 2: 选择 Prompt 模板并填充
        if self.chat_history:
            prompt = CONVERSATIONAL_RAG_PROMPT.format(
                context=context,
                chat_history=self._format_chat_history(),
                question=question,
            )
        elif use_citation:
            prompt = RAG_PROMPT_WITH_CITATION.format(
                context=context, question=question
            )
        else:
            prompt = RAG_PROMPT_TEMPLATE.format(
                context=context, question=question
            )

        # Step 3: 调用 LLM
        answer = self._call_llm(prompt)

        # Step 4: 记录对话历史
        self.chat_history.append({"role": "user", "content": question})
        self.chat_history.append({"role": "assistant", "content": answer})

        return answer

    def generate_stream(
        self,
        question: str,
        search_results: List[SearchResult],
    ) -> Generator[str, None, None]:
        """
        流式生成回答（进阶：实时返回结果）

        使用生成器逐步返回 LLM 输出的每个 token。
        适合需要实时显示的场景（如 Web UI）。

        Yields:
            每次生成的文本片段
        """
        context = self._build_context(question, search_results)
        prompt = RAG_PROMPT_WITH_CITATION.format(
            context=context, question=question
        )

        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stream=True,  # 启用流式输出
        )

        full_answer = ""
        for chunk in response:
            if chunk.choices[0].delta.content:
                text = chunk.choices[0].delta.content
                full_answer += text
                yield text

        # 记录完整的对话历史
        self.chat_history.append({"role": "user", "content": question})
        self.chat_history.append({"role": "assistant", "content": full_answer})

    # ========================================================================
    # 上下文构建
    # ========================================================================
    def _build_context(
        self,
        question: str,
        search_results: List[SearchResult],
        compress: bool = False,
    ) -> str:
        """
        构建上下文文本

        将检索结果格式化为 Prompt 中的参考资料。
        可选进行上下文压缩以去除无关内容。
        """
        if not search_results:
            return "（未找到相关参考资料）"

        context_parts = []
        for i, result in enumerate(search_results, 1):
            content = result.document.content

            # 进阶：上下文压缩
            if compress:
                content = self._compress_context(question, content)
                if content == "无关":
                    continue

            # 格式化来源信息
            source = result.document.metadata.get("filename", "未知来源")
            context_parts.append(
                f"[来源{i}] (文件: {source}, 相关度: {result.score:.2f})\n{content}"
            )

        return "\n\n".join(context_parts)

    def _compress_context(self, question: str, text: str) -> str:
        """
        上下文压缩（进阶优化）

        让 LLM 从检索到的文本中提取与问题相关的部分，
        去除无关内容，减少噪声和 token 消耗。
        """
        prompt = CONTEXT_COMPRESSION_PROMPT.format(
            question=question, text=text[:500]  # 限制长度
        )
        try:
            return self._call_llm(prompt, max_tokens=300)
        except Exception:
            return text  # 压缩失败则使用原文

    # ========================================================================
    # LLM 调用
    # ========================================================================
    def _call_llm(
        self, prompt: str, max_tokens: Optional[int] = None
    ) -> str:
        """
        调用 LLM API

        Args:
            prompt: 完整的 Prompt 文本
            max_tokens: 最大生成 token 数（可覆盖配置）

        Returns:
            LLM 生成的文本
        """
        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens,
            top_p=self.config.top_p,
        )
        return response.choices[0].message.content.strip()

    def call_llm_simple(self, prompt: str) -> str:
        """
        简单的 LLM 调用接口（供检索器等外部模块使用）

        Args:
            prompt: Prompt 文本

        Returns:
            LLM 回复
        """
        return self._call_llm(prompt)

    # ========================================================================
    # 对话管理
    # ========================================================================
    def _format_chat_history(self, max_turns: int = 5) -> str:
        """格式化最近的对话历史"""
        recent = self.chat_history[-(max_turns * 2):]
        formatted = []
        for msg in recent:
            role = "用户" if msg["role"] == "user" else "助手"
            formatted.append(f"{role}: {msg['content']}")
        return "\n".join(formatted)

    def clear_history(self):
        """清空对话历史"""
        self.chat_history = []

    def get_history(self) -> List[dict]:
        """获取对话历史"""
        return self.chat_history.copy()

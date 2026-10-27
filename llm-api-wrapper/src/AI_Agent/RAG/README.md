# RAG 知识问答系统 Demo

基于检索增强生成（Retrieval-Augmented Generation）的知识问答系统，用于学习 RAG 的核心原理与工程实践。

---

## 项目结构

```
RAG/
├── __init__.py            # 包说明与学习要点索引
├── config.py              # 全局配置（LLM、Embedding、分割、检索等参数）
├── document_loader.py     # 文档加载器（支持 txt/md/pdf/docx/json，工厂模式）
├── text_splitter.py       # 文本分割器（递归字符/句子/固定大小 三种策略）
├── embeddings.py          # 嵌入模型（API 调用/本地模型/带缓存装饰器）
├── vector_store.py        # 向量存储（FAISS + Chroma 双后端）
├── retriever.py           # 检索器（含查询改写/混合检索/MMR/重排序）
├── generator.py           # 生成器（Prompt 模板/流式输出/多轮对话/上下文压缩）
├── rag_pipeline.py        # RAG 管道（离线构建 + 在线问答 一键串联）
├── main.py                # 主入口（交互式命令行问答）
├── .env                   # API Key 配置文件（需自行填写）
├── requirements.txt       # Python 依赖列表
├── README.md              # 本文档
└── knowledge_base/        # 示例知识库文档
    ├── rag_introduction.md
    ├── rag_optimization.txt
    ├── vector_database.txt
    └── embedding_models.txt
```

### 模块职责说明

| 模块 | 职责 | 设计模式 |
|------|------|----------|
| `config.py` | 集中管理所有可配置参数，支持环境变量 | dataclass 配置聚合 |
| `document_loader.py` | 从多种格式文件提取文本，输出统一 Document 结构 | 工厂模式 |
| `text_splitter.py` | 将长文档分割为适合向量化的小块 | 策略模式 |
| `embeddings.py` | 文本向量化，支持在线 API 与本地模型 | 装饰器模式（缓存层） |
| `vector_store.py` | 向量的存储、索引与检索 | 适配器模式（统一接口） |
| `retriever.py` | 根据查询检索最相关文档，包含进阶检索策略 | 组合模式 |
| `generator.py` | 调用 LLM 基于上下文生成答案 | 模板方法 |
| `rag_pipeline.py` | 串联所有模块，提供高层 API | 管道模式 |

---

## 使用方式

### 1. 环境准备

```bash
# 安装依赖（在项目根目录 llm-api-wrapper 下）
pip install openai python-dotenv numpy faiss-cpu

# 可选依赖（按需安装）
pip install PyPDF2          # PDF 文档支持
pip install python-docx     # Word 文档支持
pip install chromadb        # Chroma 向量库支持
pip install sentence-transformers  # 本地 Embedding 模型
```

### 2. 配置 API Key

编辑 `RAG/.env` 文件：

```env
# 通义千问（推荐，性价比高）
LLM_API_KEY=your_api_key_here
LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_MODEL=qwen-plus

# Embedding 模型（可复用同一个 Key）
EMBEDDING_API_KEY=your_api_key_here
EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBEDDING_MODEL=text-embedding-v3
```

> 也可使用 OpenAI、DeepSeek 等兼容 OpenAI 接口的服务，只需修改 `BASE_URL` 和 `MODEL`。

### 3. 运行程序

```bash
# 方式一：作为模块运行（推荐，在 llm-api-wrapper 目录下）
python -m src.AI_Agent.RAG.main

# 方式二：直接运行脚本
python src/AI_Agent/RAG/main.py
```

### 4. 交互命令

启动后可使用以下命令：

| 命令 | 功能 |
|------|------|
| 直接输入问题 | 进行知识问答 |
| `/help` | 显示帮助信息 |
| `/status` | 查看知识库状态 |
| `/add` | 动态添加文本到知识库 |
| `/clear` | 清空对话历史 |
| `/stream` | 流式问答模式 |
| `/rebuild` | 重新构建知识库 |
| `/quit` | 退出程序 |

### 5. 代码方式调用

```python
from src.AI_Agent.RAG.config import RAGConfig
from src.AI_Agent.RAG.rag_pipeline import RAGPipeline

# 初始化（使用默认配置）
pipeline = RAGPipeline()

# 构建知识库
pipeline.build_knowledge_base("./knowledge_base")

# 问答
response = pipeline.query("什么是 RAG?")
print(response.format_answer())

# 流式问答
for chunk in pipeline.query_stream("向量数据库有哪些?"):
    print(chunk, end="", flush=True)

# 动态添加文档
pipeline.add_document("这是新增的知识内容...")
```

---

## RAG 核心流程

```
┌─────────────────────── 离线阶段（Indexing）───────────────────────┐
│                                                                     │
│  文档文件 → 文档加载 → 文本分割 → 向量化(Embedding) → 向量存储    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────── 在线阶段（Querying）───────────────────────┐
│                                                                     │
│  用户问题 → 查询向量化 → 相似度检索 → 上下文组装 → LLM 生成答案   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 进阶优化策略

本项目内置了多种进阶优化方法，可通过配置开启：

### 1. 查询改写（Query Rewriting）

将用户口语化的问题改写为更适合检索的形式，提升召回率。

```python
# 配置开启
config.retriever.enable_query_rewrite = True
```

示例：
- 原始查询: "RAG 怎么用"
- 改写后: "RAG 检索增强生成的使用方法和实现步骤"

### 2. 混合检索（Hybrid Search）

结合向量语义检索和关键词检索（BM25），使用 RRF 排名融合。

```python
# 配置使用混合检索
config.retriever.search_type = "hybrid"
```

优势：
- 向量检索：捕获语义相似性（"猫" ≈ "喵星人"）
- 关键词检索：精确匹配专有名词、代码片段

### 3. 最大边际相关性（MMR）

在保持相关性的同时增加结果多样性，避免返回重复内容。

```python
# 配置使用 MMR
config.retriever.search_type = "mmr"
config.retriever.mmr_lambda = 0.7  # 0=最大多样性, 1=最大相关性
```

### 4. 重排序（Reranking）

对初步检索结果使用 LLM 进行精细重排，比纯向量相似度更准确。

```python
# 配置开启重排序
config.retriever.enable_rerank = True
```

> 生产环境建议使用专门的 Rerank 模型（如 Cohere Rerank、BGE-Reranker）。

### 5. 上下文压缩（Context Compression）

对检索到的文档提取与问题相关的核心信息，去除噪声，节省 token。

```python
# 在查询时开启
response = pipeline.query("问题", compress_context=True)
```

### 6. Embedding 缓存

对相同文本的嵌入结果进行本地缓存，避免重复 API 调用。

```python
# 默认开启，缓存存储在 .embedding_cache/ 目录
embedding = create_embedding(use_cache=True)
```

### 7. 流式输出（Streaming）

实时返回 LLM 生成的每个 token，适合 Web UI 场景。

```python
for chunk in pipeline.query_stream("你的问题"):
    print(chunk, end="", flush=True)
```

### 8. 多轮对话记忆

自动维护对话历史，支持上下文连贯的追问。

```python
response1 = pipeline.query("什么是 RAG?")
response2 = pipeline.query("它和微调有什么区别?")  # 自动关联上文
pipeline.clear_history()  # 手动清空历史
```

---

## 配置参数说明

所有参数在 `config.py` 中集中管理，支持环境变量覆盖：

### 文本分割配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `chunk_size` | 500 | 每个文本块的最大字符数 |
| `chunk_overlap` | 100 | 相邻块之间的重叠字符数 |
| `strategy` | "recursive" | 分割策略：recursive / sentence / fixed |

### 检索器配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `top_k` | 5 | 返回的最相关文档数量 |
| `score_threshold` | 0.3 | 相似度阈值，低于此值会被过滤 |
| `search_type` | "similarity" | 检索策略：similarity / mmr / hybrid |
| `enable_query_rewrite` | False | 是否启用查询改写 |
| `enable_rerank` | False | 是否启用重排序 |

### 向量存储配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `store_type` | "faiss" | 向量库类型：faiss / chroma |
| `faiss_index_type` | "flat" | FAISS 索引类型：flat / ivf / hnsw |

---

## 扩展指南

### 添加新的文档格式

```python
from src.AI_Agent.RAG.document_loader import BaseLoader, DocumentLoaderFactory, Document

class CSVLoader(BaseLoader):
    def load(self, file_path: str):
        # 实现加载逻辑
        ...
        return [Document(content=text, metadata=metadata)]

# 注册到工厂
DocumentLoaderFactory.register(".csv", CSVLoader)
```

### 替换向量库

在 `config.py` 中修改：

```python
config.vector_store.store_type = "chroma"  # 切换到 Chroma
```

### 替换 LLM / Embedding 模型

修改 `.env` 文件即可切换到任何兼容 OpenAI 接口的服务：

```env
# 使用 DeepSeek
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat

# 使用 OpenAI
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
```

---

## 学习路线建议

1. **入门**: 先跑通 `main.py`，理解整体流程
2. **深入**: 阅读各模块代码和注释，理解每个环节的原理
3. **实验**: 调整 `config.py` 中的参数（chunk_size、top_k、search_type），观察效果变化
4. **进阶**: 逐步开启优化选项（query_rewrite、rerank、mmr），对比检索质量
5. **实战**: 替换为自己的知识库文档，构建领域问答系统

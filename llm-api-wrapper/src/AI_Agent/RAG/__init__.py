"""
RAG (Retrieval-Augmented Generation) 知识问答系统

项目结构:
├── config.py           # 全局配置（模型、向量库、分割参数等）
├── document_loader.py  # 文档加载器（支持 txt/pdf/md/docx 等格式）
├── text_splitter.py    # 文本分割器（多种分割策略）
├── embeddings.py       # 嵌入模型（支持多种嵌入方式）
├── vector_store.py     # 向量存储（支持 FAISS/Chroma 等）
├── retriever.py        # 检索器（含进阶检索策略）
├── generator.py        # 生成器（LLM 调用与 prompt 构建）
├── rag_pipeline.py     # RAG 管道（串联整个流程）
├── main.py             # 主入口（交互式问答演示）
├── knowledge_base/     # 示例知识库文档
└── requirements.txt    # 依赖列表

学习要点:
1. 文档加载与预处理
2. 文本分割策略（固定大小 vs 语义分割）
3. 文本向量化（Embedding）
4. 向量相似度检索
5. Prompt Engineering（上下文注入）
6. 进阶优化：重排序(Rerank)、查询改写、混合检索、上下文压缩
"""

__version__ = "1.0.0"

# 任务清单：大模型 API 封装服务

## 实施状态

| # | 任务 | 状态 | 产出文件 |
|---|------|------|---------|
| 1 | 项目初始化与配置 | ✅ 完成 | `pyproject.toml`, `.env.example`, `.gitignore` |
| 2 | 需求文档 | ✅ 完成 | `spec/requirements.md` |
| 3 | 架构设计文档 | ✅ 完成 | `spec/design.md` |
| 4 | 异常模块 | ✅ 完成 | `src/exceptions.py`, `src/models.py` |
| 5 | 工具模块 | ✅ 完成 | `src/utils/tokens.py`, `src/utils/context_trimmer.py`, `src/utils/retry.py`, `src/utils/structured_output.py` |
| 6 | 客户端模块 | ✅ 完成 | `src/client/base.py`, `src/client/qwen.py`, `src/client/deepseek.py` |
| 7 | 单元测试 | ✅ 完成 | `tests/test_tokens.py`, `tests/test_context_trimmer.py`, `tests/test_structured_output.py`, `tests/test_client.py` |
| 8 | PBT 幻觉校验 | ✅ 完成 | `tests/test_pbt_hallucination.py` |
| 9 | 任务清单 | ✅ 完成 | `spec/tasks.md`（本文件） |

## 快速开始

```bash
# 1. 安装依赖
pip install -e ".[dev]"

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 填入实际 API Key

# 3. 运行全部测试
pytest

# 4. 仅运行 PBT 测试
pytest tests/test_pbt_hallucination.py -v

# 5. 代码检查
ruff check src/ tests/
```

## 使用示例

### 基础调用

```python
import asyncio
from src.client import QwenClient
from src.models import LLMConfig

config = LLMConfig(
    api_key="your-api-key",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-turbo",
)

async def main():
    async with QwenClient(config) as client:
        response = await client.chat([
            {"role": "user", "content": "用一句话解释量子计算"}
        ])
        print(response.content)
        print(f"消耗 tokens: {response.usage.total_tokens}")

asyncio.run(main())
```

### 结构化输出

```python
from pydantic import BaseModel

class MovieReview(BaseModel):
    title: str
    rating: float
    summary: str
    pros: list[str]
    cons: list[str]

async def get_review():
    async with QwenClient(config) as client:
        review = await client.chat_structured(
            [{"role": "user", "content": "评价电影《盗梦空间》"}],
            response_model=MovieReview,
        )
        print(f"{review.title}: {review.rating}/10")
        print(f"优点: {review.pros}")
```

### 自动裁剪长上下文

```python
config = LLMConfig(
    api_key="your-key",
    base_url="https://api.deepseek.com/v1",
    model="deepseek-chat",
    max_context_tokens=4096,
    auto_trim=True,  # 超限时自动裁剪
)
```

## 后续可扩展方向

| 方向 | 说明 |
|------|------|
| 流式响应 | 实现 `stream=True` 的 `AsyncIterator[ChatChunk]` |
| 多 Provider 路由 | 基于可用性/成本/延迟的智能路由 |
| 缓存层 | 相似 prompt 命中缓存减少 API 调用 |
| 可观测性集成 | OpenTelemetry tracing / Prometheus metrics |
| summarize 裁剪策略 | 裁剪时对历史对话生成摘要而非直接丢弃 |
| 批量请求 | 支持 batch API 接口降低成本 |

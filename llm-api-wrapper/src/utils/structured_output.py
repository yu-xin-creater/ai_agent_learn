"""结构化 JSON 输出工具

将 Pydantic BaseModel schema 注入 prompt，解析并校验模型响应。
"""

from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from src.exceptions import LLMResponseError

T = TypeVar("T", bound=BaseModel)


def generate_schema_prompt(model_class: type[BaseModel]) -> str:
    """根据 Pydantic 模型生成 JSON Schema 描述，用于注入 system prompt

    Args:
        model_class: Pydantic BaseModel 子类

    Returns:
        供 LLM 理解的 schema 描述文本
    """
    schema = model_class.model_json_schema()
    schema_str = json.dumps(schema, ensure_ascii=False, indent=2)

    return (
        "你必须严格按照以下 JSON Schema 格式输出，不要输出任何其他内容：\n"
        f"```json\n{schema_str}\n```\n"
        "请直接输出合法的 JSON 对象，不要包含 markdown 代码块标记或其他文字说明。"
    )


def inject_schema_into_messages(
    messages: list[dict[str, str]],
    model_class: type[BaseModel],
) -> list[dict[str, str]]:
    """将 schema 描述注入到消息列表中

    策略：如果已有 system 消息，追加 schema 描述；否则新建 system 消息。

    Args:
        messages: 原始消息列表
        model_class: 期望输出的 Pydantic 模型类

    Returns:
        注入 schema 后的消息列表（新列表，不修改原始）
    """
    schema_prompt = generate_schema_prompt(model_class)
    result = list(messages)

    # 查找已有的 system 消息
    system_idx = None
    for i, msg in enumerate(result):
        if msg.get("role") == "system":
            system_idx = i
            break

    if system_idx is not None:
        # 追加到现有 system 消息
        original_content = result[system_idx].get("content", "")
        result[system_idx] = {
            "role": "system",
            "content": f"{original_content}\n\n{schema_prompt}",
        }
    else:
        # 新建 system 消息插入到最前面
        result.insert(0, {"role": "system", "content": schema_prompt})

    return result


def extract_json_from_response(text: str) -> str:
    """从模型响应中提取 JSON 字符串

    处理模型可能返回 markdown 代码块包裹的 JSON。

    Args:
        text: 模型原始响应文本

    Returns:
        提取出的 JSON 字符串

    Raises:
        LLMResponseError: 无法提取有效 JSON
    """
    text = text.strip()

    # 尝试直接解析
    if text.startswith("{") or text.startswith("["):
        return text

    # 尝试从 markdown 代码块中提取
    pattern = r"```(?:json)?\s*\n?([\s\S]*?)\n?```"
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()

    # 尝试找到第一个 { 到最后一个 }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]

    raise LLMResponseError(
        "无法从响应中提取 JSON",
        raw_response=text,
    )


def parse_structured_response(
    text: str,
    model_class: type[T],
) -> T:
    """解析并校验结构化响应

    Args:
        text: 模型响应文本
        model_class: 期望的 Pydantic 模型类

    Returns:
        校验通过的模型实例

    Raises:
        LLMResponseError: JSON 解析失败或 schema 校验失败
    """
    json_str = extract_json_from_response(text)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise LLMResponseError(
            f"JSON 解析失败: {e}",
            raw_response=text,
        ) from e

    try:
        return model_class.model_validate(data)
    except ValidationError as e:
        errors = e.errors()
        raise LLMResponseError(
            f"结构化输出校验失败: {len(errors)} 个错误",
            validation_errors=[
                {"loc": err["loc"], "msg": err["msg"], "type": err["type"]}
                for err in errors
            ],
            raw_response=text,
        ) from e


def build_validation_feedback(error: LLMResponseError) -> str:
    """根据校验错误构建反馈 prompt，用于重试时告知模型修正

    Args:
        error: 包含 validation_errors 的响应异常

    Returns:
        反馈文本，追加到下一次请求的 user 消息中
    """
    if not error.validation_errors:
        return f"你的上一次输出格式不正确：{error.message}。请严格按照要求的 JSON Schema 重新输出。"

    feedback_lines = ["你的上一次输出不符合要求的 JSON Schema，请修正以下问题："]
    for err in error.validation_errors:
        loc = " → ".join(str(x) for x in err["loc"])
        feedback_lines.append(f"  - 字段 `{loc}`: {err['msg']}")
    feedback_lines.append("\n请严格按照要求的 JSON Schema 重新输出合法 JSON。")
    return "\n".join(feedback_lines)

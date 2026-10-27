"""结构化输出模块测试"""

import pytest
from pydantic import BaseModel

from src.exceptions import LLMResponseError
from src.utils.structured_output import (
    build_validation_feedback,
    extract_json_from_response,
    generate_schema_prompt,
    inject_schema_into_messages,
    parse_structured_response,
)


class UserInfo(BaseModel):
    """测试用 Pydantic 模型"""

    name: str
    age: int
    city: str


class NestedModel(BaseModel):
    """嵌套模型"""

    title: str
    items: list[str]
    metadata: dict[str, int]


class TestGenerateSchemaPrompt:
    def test_contains_schema(self):
        prompt = generate_schema_prompt(UserInfo)
        assert "name" in prompt
        assert "age" in prompt
        assert "city" in prompt
        assert "JSON Schema" in prompt

    def test_contains_type_info(self):
        prompt = generate_schema_prompt(UserInfo)
        assert "string" in prompt or "str" in prompt

    def test_nested_model(self):
        prompt = generate_schema_prompt(NestedModel)
        assert "title" in prompt
        assert "items" in prompt
        assert "metadata" in prompt


class TestInjectSchemaIntoMessages:
    def test_inject_with_existing_system(self):
        messages = [
            {"role": "system", "content": "你是助手"},
            {"role": "user", "content": "问题"},
        ]
        result = inject_schema_into_messages(messages, UserInfo)
        assert result[0]["role"] == "system"
        assert "你是助手" in result[0]["content"]
        assert "JSON Schema" in result[0]["content"]

    def test_inject_without_system(self):
        messages = [{"role": "user", "content": "问题"}]
        result = inject_schema_into_messages(messages, UserInfo)
        assert result[0]["role"] == "system"
        assert "JSON Schema" in result[0]["content"]
        assert result[1] == messages[0]

    def test_does_not_mutate_original(self):
        messages = [{"role": "user", "content": "问题"}]
        original_len = len(messages)
        inject_schema_into_messages(messages, UserInfo)
        assert len(messages) == original_len


class TestExtractJsonFromResponse:
    def test_plain_json(self):
        text = '{"name": "test", "age": 1, "city": "bj"}'
        result = extract_json_from_response(text)
        assert result == text

    def test_json_in_markdown_block(self):
        text = '```json\n{"name": "test", "age": 1, "city": "bj"}\n```'
        result = extract_json_from_response(text)
        assert '"name"' in result

    def test_json_in_generic_code_block(self):
        text = '```\n{"name": "test"}\n```'
        result = extract_json_from_response(text)
        assert '"name"' in result

    def test_json_with_surrounding_text(self):
        text = '以下是结果：\n{"name": "test", "age": 1, "city": "bj"}\n以上。'
        result = extract_json_from_response(text)
        assert '"name"' in result

    def test_no_json_raises_error(self):
        text = "这里没有任何 JSON 内容"
        with pytest.raises(LLMResponseError):
            extract_json_from_response(text)

    def test_array_json(self):
        text = '[{"a": 1}, {"b": 2}]'
        result = extract_json_from_response(text)
        assert result == text


class TestParseStructuredResponse:
    def test_valid_response(self):
        text = '{"name": "张三", "age": 25, "city": "北京"}'
        result = parse_structured_response(text, UserInfo)
        assert isinstance(result, UserInfo)
        assert result.name == "张三"
        assert result.age == 25
        assert result.city == "北京"

    def test_invalid_json(self):
        text = "not json at all"
        with pytest.raises(LLMResponseError):
            parse_structured_response(text, UserInfo)

    def test_schema_validation_failure(self):
        """缺少必要字段"""
        text = '{"name": "张三"}'
        with pytest.raises(LLMResponseError) as exc_info:
            parse_structured_response(text, UserInfo)
        assert exc_info.value.validation_errors is not None
        assert len(exc_info.value.validation_errors) > 0

    def test_wrong_type(self):
        """类型错误"""
        text = '{"name": "张三", "age": "not_a_number", "city": "北京"}'
        with pytest.raises(LLMResponseError) as exc_info:
            parse_structured_response(text, UserInfo)
        assert exc_info.value.validation_errors is not None

    def test_nested_model_valid(self):
        text = '{"title": "test", "items": ["a", "b"], "metadata": {"x": 1}}'
        result = parse_structured_response(text, NestedModel)
        assert result.title == "test"
        assert result.items == ["a", "b"]

    def test_json_in_markdown(self):
        """模型用 markdown 包裹 JSON 时也能正确解析"""
        text = '```json\n{"name": "李四", "age": 30, "city": "上海"}\n```'
        result = parse_structured_response(text, UserInfo)
        assert result.name == "李四"


class TestBuildValidationFeedback:
    def test_with_validation_errors(self):
        error = LLMResponseError(
            "校验失败",
            validation_errors=[
                {"loc": ("age",), "msg": "Input should be a valid integer", "type": "int_parsing"},
            ],
        )
        feedback = build_validation_feedback(error)
        assert "age" in feedback
        assert "JSON Schema" in feedback

    def test_without_validation_errors(self):
        error = LLMResponseError("格式不正确")
        feedback = build_validation_feedback(error)
        assert "JSON Schema" in feedback
        assert "格式不正确" in feedback

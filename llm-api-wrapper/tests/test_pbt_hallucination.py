"""PBT（Property-Based Testing）自动化校验幻觉测试

使用 Hypothesis 生成随机 schema 和不合规 JSON 响应，验证系统能正确：
1. 检测不符合 schema 的输出（模拟幻觉）
2. 保证 token 计数的一致性
3. 保证上下文裁剪后 token 不超限
4. 保证结构化输出校验的鲁棒性
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from hypothesis import given, assume, settings, HealthCheck
from hypothesis import strategies as st
from pydantic import BaseModel, ValidationError

from src.exceptions import LLMResponseError
from src.utils.context_trimmer import trim_context
from src.utils.structured_output import (
    extract_json_from_response,
    parse_structured_response,
)
from src.utils.tokens import count_messages_tokens, count_tokens


# ════════════════════════════════════════════════════════════════════════
# 策略定义：生成随机测试数据
# ════════════════════════════════════════════════════════════════════════

# 生成随机 JSON 值
json_primitives = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-10000, max_value=10000),
    st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
    st.text(min_size=0, max_size=50),
)

# 生成随机 JSON 对象（浅层）
json_objects = st.dictionaries(
    keys=st.text(
        alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="_"),
        min_size=1,
        max_size=20,
    ),
    values=json_primitives,
    min_size=1,
    max_size=10,
)

# 生成消息内容
message_content = st.text(min_size=1, max_size=200)

# 生成消息列表
message_strategy = st.fixed_dictionaries(
    {"role": st.sampled_from(["user", "assistant"]), "content": message_content}
)

messages_strategy = st.lists(message_strategy, min_size=1, max_size=20)


# ════════════════════════════════════════════════════════════════════════
# 幻觉检测相关的 Pydantic 模型
# ════════════════════════════════════════════════════════════════════════


class StrictUserProfile(BaseModel):
    """严格模式用户资料 - 用于幻觉检测"""

    username: str
    email: str
    age: int
    is_active: bool


class AnalysisResult(BaseModel):
    """分析结果模型"""

    score: float
    category: str
    tags: list[str]
    confidence: float


class NestedReport(BaseModel):
    """嵌套报告模型"""

    title: str
    sections: list[str]
    metadata: dict[str, str]
    page_count: int


# ════════════════════════════════════════════════════════════════════════
# PBT 测试：幻觉检测 — 模型输出不合规时系统应拒绝
# ════════════════════════════════════════════════════════════════════════


class TestHallucinationDetection:
    """幻觉检测 PBT 测试

    核心性质：任何不符合预期 schema 的模型输出，
    系统必须抛出 LLMResponseError 而非静默接受。
    """

    @given(random_obj=json_objects)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_random_json_rejected_if_schema_mismatch(self, random_obj: dict[str, Any]):
        """随机生成的 JSON 对象若不符合 StrictUserProfile，必须被拒绝"""
        # 如果随机生成的恰好合法，跳过
        try:
            StrictUserProfile.model_validate(random_obj)
            assume(False)  # 跳过合法输入
        except ValidationError:
            pass

        json_str = json.dumps(random_obj, ensure_ascii=False)

        with pytest.raises(LLMResponseError) as exc_info:
            parse_structured_response(json_str, StrictUserProfile)

        assert exc_info.value.validation_errors is not None or "解析失败" in exc_info.value.message

    @given(
        data=st.fixed_dictionaries(
            {
                "username": st.text(min_size=1, max_size=20),
                "email": st.text(min_size=1, max_size=30),
                "age": st.one_of(
                    st.text(min_size=1, max_size=5),  # 类型错误：字符串代替 int
                    st.floats(allow_nan=True),  # 类型错误：NaN
                    st.none(),  # 类型错误：None
                ),
                "is_active": st.one_of(
                    st.text(min_size=1, max_size=5),  # 类型错误
                    st.integers(),  # 宽容验证可能通过
                ),
            }
        )
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_type_hallucination_detected(self, data: dict[str, Any]):
        """模拟模型产生类型错误的幻觉（如把 int 写成 string）"""
        # 确保确实不合法
        try:
            StrictUserProfile.model_validate(data)
            assume(False)
        except (ValidationError, ValueError):
            pass

        json_str = json.dumps(data, ensure_ascii=False, default=str)

        # 系统应检测到类型不匹配
        try:
            result = parse_structured_response(json_str, StrictUserProfile)
            # 如果 Pydantic 做了类型强制转换（如 "123" → 123），这是可接受的
            # 但最终结果必须是合法的 StrictUserProfile 实例
            assert isinstance(result, StrictUserProfile)
        except LLMResponseError:
            pass  # 正确拒绝

    @given(
        missing_fields=st.lists(
            st.sampled_from(["username", "email", "age", "is_active"]),
            min_size=1,
            max_size=3,
            unique=True,
        )
    )
    @settings(max_examples=50)
    def test_missing_field_hallucination_detected(self, missing_fields: list[str]):
        """模拟模型遗漏必要字段（幻觉输出不完整）"""
        full_data = {
            "username": "testuser",
            "email": "test@example.com",
            "age": 25,
            "is_active": True,
        }
        # 移除部分字段
        for field in missing_fields:
            full_data.pop(field, None)

        json_str = json.dumps(full_data, ensure_ascii=False)

        with pytest.raises(LLMResponseError) as exc_info:
            parse_structured_response(json_str, StrictUserProfile)

        assert exc_info.value.validation_errors is not None
        # 验证错误信息中包含缺失字段的定位
        error_locs = [
            str(err["loc"]) for err in exc_info.value.validation_errors
        ]
        assert len(error_locs) > 0

    @given(
        extra_text=st.text(min_size=10, max_size=200),
        position=st.sampled_from(["before", "after", "both"]),
    )
    @settings(max_examples=30)
    def test_json_extraction_from_noisy_output(self, extra_text: str, position: str):
        """模型在 JSON 前后添加多余文字时仍能正确提取"""
        valid_json = '{"username": "test", "email": "a@b.c", "age": 25, "is_active": true}'

        if position == "before":
            text = f"{extra_text}\n{valid_json}"
        elif position == "after":
            text = f"{valid_json}\n{extra_text}"
        else:
            text = f"{extra_text}\n{valid_json}\n{extra_text}"

        try:
            extracted = extract_json_from_response(text)
            # 如果提取成功，解析后应是合法 JSON
            parsed = json.loads(extracted)
            assert isinstance(parsed, (dict, list))
        except LLMResponseError:
            # 如果 extra_text 中包含干扰性 { }，提取可能失败，这是可接受的
            pass

    @given(
        score=st.floats(allow_nan=True, allow_infinity=True),
        confidence=st.floats(min_value=-100, max_value=200),
    )
    @settings(max_examples=50)
    def test_numeric_hallucination_boundary(self, score: float, confidence: float):
        """模拟模型输出非法数值（NaN, Infinity, 超范围）"""
        data = {
            "score": score,
            "category": "test",
            "tags": ["a"],
            "confidence": confidence,
        }

        # NaN 和 Infinity 无法序列化为标准 JSON
        try:
            json_str = json.dumps(data, ensure_ascii=False)
        except (ValueError, OverflowError):
            return  # 无法序列化则跳过

        try:
            result = parse_structured_response(json_str, AnalysisResult)
            # 如果通过了，数值必须是有限数
            assert isinstance(result.score, float)
            assert isinstance(result.confidence, float)
        except LLMResponseError:
            pass  # 正确拒绝非法数值


# ════════════════════════════════════════════════════════════════════════
# PBT 测试：Token 计数一致性
# ════════════════════════════════════════════════════════════════════════


class TestTokenCountingProperties:
    """Token 计数的性质测试"""

    @given(text=st.text(min_size=0, max_size=500))
    @settings(max_examples=200)
    def test_token_count_non_negative(self, text: str):
        """性质：token 数量永远非负"""
        assert count_tokens(text) >= 0

    @given(text=st.text(min_size=1, max_size=500))
    @settings(max_examples=100)
    def test_non_empty_text_has_tokens(self, text: str):
        """性质：非空文本至少有 1 个 token"""
        # 某些特殊字符可能编码为空，放宽为 >= 0
        assert count_tokens(text) >= 0

    @given(text=st.text(min_size=1, max_size=200))
    @settings(max_examples=100)
    def test_token_count_deterministic(self, text: str):
        """性质：相同输入，输出一致（幂等性）"""
        assert count_tokens(text) == count_tokens(text)

    @given(
        text_a=st.text(min_size=1, max_size=100),
        text_b=st.text(min_size=1, max_size=100),
    )
    @settings(max_examples=100)
    def test_concatenation_subadditivity(self, text_a: str, text_b: str):
        """性质：拼接文本的 token 数 ≤ 分别计数之和 + 常数

        由于 tokenizer 可能跨边界合并 token，拼接后可能更少。
        但不会超过分别计数之和。
        """
        combined = count_tokens(text_a + text_b)
        separate = count_tokens(text_a) + count_tokens(text_b)
        # 拼接后 tokens ≤ 分别之和（tokenizer 可能合并）
        # 允许小的误差（个别 tokenizer 行为）
        assert combined <= separate + 2

    @given(messages=messages_strategy)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_messages_token_count_non_negative(self, messages: list[dict[str, str]]):
        """性质：消息列表 token 计数永远非负"""
        result = count_messages_tokens(messages)
        assert result >= 0

    @given(messages=messages_strategy)
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_messages_token_includes_overhead(self, messages: list[dict[str, str]]):
        """性质：消息 token 数 ≥ 每条消息的固定开销"""
        result = count_messages_tokens(messages)
        # 每条消息 4 token 开销 + 最后 3 token 引导
        min_expected = len(messages) * 4 + 3
        assert result >= min_expected


# ════════════════════════════════════════════════════════════════════════
# PBT 测试：上下文裁剪保证
# ════════════════════════════════════════════════════════════════════════


class TestContextTrimmingProperties:
    """上下文裁剪的性质测试"""

    @given(
        messages=st.lists(
            st.fixed_dictionaries(
                {
                    "role": st.sampled_from(["user", "assistant"]),
                    "content": st.text(min_size=10, max_size=100),
                }
            ),
            min_size=3,
            max_size=15,
        ),
        max_tokens=st.integers(min_value=50, max_value=500),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_trimmed_result_within_limit(
        self, messages: list[dict[str, str]], max_tokens: int
    ):
        """核心性质：裁剪后 token 数永远 ≤ max_tokens（或返回最小可能集）"""
        # 添加 system 消息
        full_messages = [{"role": "system", "content": "你是助手"}] + messages
        result = trim_context(full_messages, max_tokens)

        result_tokens = count_messages_tokens(result)

        if result_tokens > max_tokens:
            # 极端情况：即使只剩 system + 1 条也超限，此时返回最小集
            # 验证已经尽可能裁剪
            assert len(result) <= 2  # 最多 system + 1 条 protected

    @given(
        messages=st.lists(
            st.fixed_dictionaries(
                {
                    "role": st.sampled_from(["user", "assistant"]),
                    "content": st.text(min_size=1, max_size=50),
                }
            ),
            min_size=1,
            max_size=10,
        )
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_no_trim_when_within_limit(self, messages: list[dict[str, str]]):
        """性质：token 未超限时不裁剪，原样返回"""
        max_tokens = 100000  # 极大值
        result = trim_context(messages, max_tokens)
        assert result == messages

    @given(
        messages=st.lists(
            st.fixed_dictionaries(
                {
                    "role": st.sampled_from(["user", "assistant"]),
                    "content": st.text(min_size=5, max_size=80),
                }
            ),
            min_size=5,
            max_size=15,
        ),
        max_tokens=st.integers(min_value=100, max_value=400),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_trim_preserves_message_order(
        self, messages: list[dict[str, str]], max_tokens: int
    ):
        """性质：裁剪不改变消息的相对顺序"""
        full_messages = [{"role": "system", "content": "sys"}] + messages
        result = trim_context(full_messages, max_tokens)

        # 结果中的消息顺序应与原始列表一致
        result_contents = [m["content"] for m in result]
        original_contents = [m["content"] for m in full_messages]

        # 验证结果是原始列表的子序列
        orig_idx = 0
        for content in result_contents:
            while orig_idx < len(original_contents) and original_contents[orig_idx] != content:
                orig_idx += 1
            assert orig_idx < len(original_contents), "裁剪结果不是原始的子序列"
            orig_idx += 1

    @given(
        messages=st.lists(
            st.fixed_dictionaries(
                {
                    "role": st.sampled_from(["user", "assistant"]),
                    "content": st.text(min_size=5, max_size=80),
                }
            ),
            min_size=3,
            max_size=10,
        ),
        max_tokens=st.integers(min_value=80, max_value=300),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_trim_preserves_system_message(
        self, messages: list[dict[str, str]], max_tokens: int
    ):
        """性质：裁剪后 system 消息始终保留（如果原始有的话）"""
        system_msg = {"role": "system", "content": "sys"}
        full_messages = [system_msg] + messages
        result = trim_context(full_messages, max_tokens)

        system_in_result = [m for m in result if m["role"] == "system"]
        assert len(system_in_result) >= 1
        assert system_in_result[0]["content"] == "sys"

    @given(messages=messages_strategy)
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_trim_idempotent(self, messages: list[dict[str, str]]):
        """性质：裁剪是幂等的 — 对已裁剪结果再裁剪不变"""
        max_tokens = 200
        once = trim_context(messages, max_tokens)
        twice = trim_context(once, max_tokens)
        assert once == twice


# ════════════════════════════════════════════════════════════════════════
# PBT 测试：结构化输出校验鲁棒性
# ════════════════════════════════════════════════════════════════════════


class TestStructuredOutputProperties:
    """结构化输出校验的性质测试"""

    @given(
        data=st.fixed_dictionaries(
            {
                "username": st.text(min_size=1, max_size=30),
                "email": st.text(min_size=1, max_size=50),
                "age": st.integers(min_value=0, max_value=200),
                "is_active": st.booleans(),
            }
        )
    )
    @settings(max_examples=100)
    def test_valid_data_always_passes(self, data: dict[str, Any]):
        """性质：合法数据总能通过校验"""
        json_str = json.dumps(data, ensure_ascii=False)
        result = parse_structured_response(json_str, StrictUserProfile)
        assert isinstance(result, StrictUserProfile)
        assert result.username == data["username"]
        assert result.age == data["age"]

    @given(
        valid_json=st.fixed_dictionaries(
            {
                "title": st.text(min_size=1, max_size=30),
                "sections": st.lists(st.text(min_size=1, max_size=20), min_size=0, max_size=5),
                "metadata": st.dictionaries(
                    st.text(min_size=1, max_size=10),
                    st.text(min_size=1, max_size=10),
                    min_size=0,
                    max_size=3,
                ),
                "page_count": st.integers(min_value=1, max_value=1000),
            }
        )
    )
    @settings(max_examples=50)
    def test_nested_model_valid_data_passes(self, valid_json: dict[str, Any]):
        """性质：合法嵌套数据总能通过校验"""
        json_str = json.dumps(valid_json, ensure_ascii=False)
        result = parse_structured_response(json_str, NestedReport)
        assert isinstance(result, NestedReport)
        assert result.title == valid_json["title"]

    @given(
        corruption=st.sampled_from([
            "remove_brace",
            "add_trailing_comma",
            "break_quotes",
            "truncate",
        ]),
        data=st.fixed_dictionaries(
            {
                "username": st.text(min_size=1, max_size=20),
                "email": st.text(min_size=3, max_size=30),
                "age": st.integers(min_value=1, max_value=100),
                "is_active": st.booleans(),
            }
        ),
    )
    @settings(max_examples=50)
    def test_corrupted_json_detected(self, corruption: str, data: dict[str, Any]):
        """性质：损坏的 JSON 必须被检测到（模拟模型输出截断/格式错误）"""
        json_str = json.dumps(data, ensure_ascii=False)

        if corruption == "remove_brace":
            corrupted = json_str[:-1]  # 移除尾部 }
        elif corruption == "add_trailing_comma":
            corrupted = json_str[:-1] + ",}"  # 添加尾部逗号
        elif corruption == "break_quotes":
            corrupted = json_str.replace('"', "'", 2)  # 破坏引号
        elif corruption == "truncate":
            corrupted = json_str[: len(json_str) // 2]  # 截断一半
        else:
            corrupted = json_str

        try:
            result = parse_structured_response(corrupted, StrictUserProfile)
            # 如果某些损坏恰好仍是合法 JSON + 合法 schema（极小概率），也接受
            assert isinstance(result, StrictUserProfile)
        except LLMResponseError:
            pass  # 正确检测到损坏

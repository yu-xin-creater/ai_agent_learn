"""翻译工具（模拟实现）"""

from langchain_core.tools import tool


@tool
def translate(text: str, target_language: str = "英文") -> str:
    """
    将中文文本翻译成其他语言（模拟）

    Args:
        text: 要翻译的中文文本
        target_language: 目标语言，支持"英文"、"日文"、"韩文"
    """
    translations = {
        "你好": {"英文": "Hello", "日文": "こんにちは", "韩文": "안녕하세요"},
        "谢谢": {"英文": "Thank you", "日文": "ありがとう", "韩文": "감사합니다"},
        "再见": {"英文": "Goodbye", "日文": "さようなら", "韩文": "안녕히 가세요"},
    }

    if text in translations and target_language in translations[text]:
        result = translations[text][target_language]
        return f"翻译结果：'{text}' → [{target_language}] {result}"

    return f"暂不支持'{text}'的{target_language}翻译"

from langchain_core.tools import tool
from datetime import datetime

@tool
def get_weather(city: str) -> str:
    """
    获取指定城市的实时天气信息
    
    Args:
        city: 城市名称，支持北京、上海、广州、深圳、杭州等
    """
    weather_database = {
        "北京": {"condition": "晴天", "temp": "15-25°C", "aqi": "优"},
        "上海": {"condition": "多云", "temp": "18-28°C", "aqi": "良"},
        "深圳": {"condition": "小雨", "temp": "22-30°C", "aqi": "优"},
        "杭州": {"condition": "阴天", "temp": "17-26°C", "aqi": "良"},
        "广州": {"condition": "晴天", "temp": "20-32°C", "aqi": "良"},
    }
    
    if city not in weather_database:
        return f"暂无{city}的天气数据，支持城市：北京、上海、深圳、杭州、广州"
    
    data = weather_database[city]
    return f"""
{city} 天气预报
━━━━━━━━━━━━━━━━
天气：{data['condition']}
温度：{data['temp']}
空气质量：{data['aqi']}
更新时间：{datetime.now().strftime("%H:%M")}
    """.strip()
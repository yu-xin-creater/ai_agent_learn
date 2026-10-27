from mcp.server.fastmcp import FastMCP

# 初始化 MCP 服务器
mcp = FastMCP("WeatherServer")

@mcp.tool()  # 装饰器将函数注册为 MCP 工具
async def query_weather(city: str) -> str:
    """
    输入指定城市的英文名称，返回今日天气查询结果。
    :param city: 城市名称（需使用英文）
    :return: 格式化后的天气信息
    """
    data = await fetch_weather(city)  # 调用 OpenWeather API
    return format_weather(data)       # 格式化返回

@mcp.tool()
async def get_weather_tips(season: str) -> str:
    """
    获取指定季节的天气贴士。
    :param season: 季节名称 (spring, summer, autumn, winter)
    """
    tips = {
        "spring": "🌸 春季多风，注意防风保暖",
        "summer": "☀️ 夏季炎热，注意防暑",
        # ...
    }
    return tips.get(season.lower(), "❓ 未知季节")

if __name__ == "__main__":
    mcp.run(transport='stdio')  # 以 STDIO 模式启动
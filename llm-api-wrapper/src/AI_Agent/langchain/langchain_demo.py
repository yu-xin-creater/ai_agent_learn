from dotenv import load_dotenv
import os

load_dotenv()

# 通义千问示例
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import PromptTemplate # 简单模板
from langchain_core.prompts import ChatPromptTemplate # 聊天模板


## 1、定义模板 Prompt Template
# template = PromptTemplate.from_template(
#     "请用{language}语言介绍一下{topic}，不超过100字。"
# )

# # 填充变量
# prompt = template.format(language="中文", topic="人工智能")

# # 定义多轮对话模板
# chat_template = ChatPromptTemplate.from_messages([
#     ("system", "你是一位{role}，擅长用简洁易懂的方式解释复杂概念。"),
#     ("human", "请解释一下：{concept}"),
# ])

# # 生成消息列表
# messages = chat_template.format_messages(
#     role="物理学教授",
#     concept="量子纠缠"
# )

# # 定义模型
# llm = ChatTongyi(
#     model="qwen-turbo",
#     dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
# )

# # 调用模型
# response = llm.invoke(messages)
# print(response.content)

#==================================================================#

## 2、Tools工具
from langchain_core.tools import tool
from typing import Optional

# #2.1 装饰器
# @tool
# def get_weather(city: str) -> str:
#     # 重点：docstring 写清楚！ AI 就是靠这个描述来决定什么时候调用这个工具。
#     """
#     获取城市的天气信息
    
#     Args:
#         city: 城市名称，如"北京"、"上海"
#     """
#     # 模拟天气数据（实际项目中调用天气 API）
#     weather_data = {
#         "北京": "晴天，25°C",
#         "上海": "多云，22°C",
#         "深圳": "小雨，28°C",
#     }
#     return weather_data.get(city, f"{city}的天气暂时无法获取")


# # 测试工具
# result = get_weather.invoke({"city": "北京"})
# print(result)  # 输出: 晴天，25°C    

# # 2.2 更复杂的工具
# @tool
# def search_database(query: str, category: Optional[str] = None) -> str:
#     """
#     在数据库中搜索信息
    
#     Args:
#         query: 搜索关键词
#         category: 可选的分类过滤器，如"科技"、"健康"、"教育"
#     """
#     database = {
#         "科技": ["人工智能正在改变世界", "5G技术的应用"],
#         "健康": ["健康饮食的重要性", "运动与长寿的关系"],
#         "教育": ["在线学习的趋势", "终身学习的价值"]
#     }
    
#     results = []
    
#     if category and category in database:
#         for item in database[category]:
#             if query.lower() in item.lower():
#                 results.append(f"[{category}] {item}")
#     else:
#         for cat, items in database.items():
#             for item in items:
#                 if query.lower() in item.lower():
#                     results.append(f"[{cat}] {item}")
    
#     if results:
#         return "找到以下结果:\n" + "\n".join(results)
#     else:
#         return f"没有找到关于 '{query}' 的结果"

# 2.3 用类定义
  

#==================================================================#
from langchain.agents import create_agent
import os
from langgraph.checkpoint.memory import MemorySaver # 短期记忆
# 1、初始化模型
llm = ChatTongyi(
    model="qwen-plus",
    temperature=0.7,
    dashscope_api_key=os.environ.get("DASHSCOPE_API_KEY")
)
# 2、定义工具
@tool
def get_weather(city: str) -> str:
    """获取城市天气信息"""
    weather_db = {
        "北京": "晴天，15-25度",
        "上海": "多云，18-28度",
        "深圳": "小雨，20-30度",
    }
    return weather_db.get(city, f"{city}的天气信息暂不可用")

@tool
def calculator(expression: str) -> str:
    """执行数学计算"""
    try:
        result = eval(expression)
        return f"计算结果: {expression} = {result}"
    except:
        return "计算错误"

@tool  
def search_knowledge(query: str) -> str:
    """搜索知识库"""
    knowledge = {
        "LangChain": "LangChain是一个用于开发LLM应用的框架，支持工具、代理、内存管理等功能。",
        "机器学习": "机器学习是AI的子集，让系统能从数据中自动学习和改进。",
    }
    for key, value in knowledge.items():
        if key in query:
            return value
    return f"未找到关于'{query}'的信息"


# 创建记忆存储
checkpointer = MemorySaver()

# 3、创建Agent
agent = create_agent(
    model=llm,
    tools=[get_weather, calculator, search_knowledge],
    system_prompt="你是一个专业的中文助手。仔细分析用户问题，选择合适的工具来回答。",
    checkpointer=checkpointer  # 启用记忆
)

# 会话配置（同一个 thread_id 共享记忆）
config = {"configurable": {"thread_id": "session_001"}}



# 4、使用Agent
result = agent.invoke({
    "messages": [{"role": "user", "content": "北京今天天气怎么样？"}]
},
config
)

# 获取回答
final_message = result["messages"][-1]
print(f"回答: {final_message.content}")

# 流式调用
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "北京天气如何？"}]},
    config,
    stream_mode="values"
):
    messages = chunk.get("messages")
    if messages:
        last_message = messages[-1]
        if hasattr(last_message, "content") and last_message.content:
            print(last_message.content, end="", flush=True)
print()  # 换行


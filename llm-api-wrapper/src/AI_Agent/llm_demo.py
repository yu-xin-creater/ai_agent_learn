from langchain_openai import ChatOpenAI

# 初始化LLM作为agent的大脑
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0, # 降低随机性，提高推理稳定性
)


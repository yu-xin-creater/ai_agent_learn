from fastapi import FastAPI
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

# 全局变量
mcp_client: MultiServerMCPClient = None
agent = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 生命周期管理"""
    global mcp_client, agent
    
    # 1. 读取 MCP 服务器配置
    servers_cfg = Configuration.load_servers()
    
    # 2. 连接 MCP 服务器并获取工具
    mcp_client = MultiServerMCPClient(servers_cfg)
    tools = await mcp_client.get_tools()
    
    # 3. 初始化 ReAct Agent
    model = ChatTongyi(model=cfg.model, streaming=False)
    checkpointer = InMemorySaver()  # 内存中保存对话历史
    
    agent = create_react_agent(
        model=model, 
        tools=tools,
        prompt=prompt,
        checkpointer=checkpointer
    )
    
    yield  # 应用运行中
    
    # 4. 清理资源
    await mcp_client.cleanup()

app = FastAPI(lifespan=lifespan)

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """聊天接口"""
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=request.message)]},
        {"configurable": {"thread_id": request.thread_id}}
    )
    return ChatResponse(content=result["messages"][-1].content)
async def run_chat_loop() -> None:
    """启动 MCP-Agent 聊天循环"""
    
    # 1. 连接多台 MCP 服务器
    mcp_client = MultiServerMCPClient(servers_cfg)
    tools = await mcp_client.get_tools()
    
    # 2. 初始化大模型
    model = ChatTongyi(model=cfg.model)
    
    # 3. 构造 Agent（带记忆）
    checkpointer = InMemorySaver()
    agent = create_react_agent(
        model=model, 
        tools=tools,
        prompt=prompt,
        checkpointer=checkpointer
    )
    
    # 4. CLI 聊天循环
    while True:
        user_input = input("\n你: ").strip()
        if user_input.lower() == "quit":
            break
            
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config={"configurable": {"thread_id": "1"}}  # 对话ID
        )
        print(f"\nAI: {result['messages'][-1].content}")
    
    # 5. 清理
    await mcp_client.cleanup()
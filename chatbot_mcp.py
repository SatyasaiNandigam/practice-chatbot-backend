from typing import TypedDict, Annotated, List

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
import sqlite3
from langchain.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchResults
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

    
llm = ChatOpenAI(model="gpt-4o-mini")

client = MultiServerMCPClient(
    {
        "arith": {
            "transport": "stdio",
            "command": "Python3",
            "args" : ["/Users/Satya/Desktop/mcp-math-server/main.py"]
        },
        "expense": {
            "transport": "streamable_http",
            "url": "https://splendind-gold-dingo.fastmcp.app/mcp"
        }
    }
)



class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

async def build_graph():

    tools = await client.get_tools()
    
    llm_with_tools = llm.bind_tools(tools)

    graph = StateGraph(ChatState)
    
    async def chat_node(state: ChatState) -> ChatState:
        messages = state["messages"]
        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}

    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", ToolNode(tools))
    graph.set_entry_point("chat_node")
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")



    chatbot = graph.compile()
    
    return chatbot

async def main():
    chatbot = await build_graph()
    
    response = await chatbot.ainvoke({"messages": [HumanMessage(content="Find the modulus of 132354 and 23 and give answer like a cricket commentator.")]})
    
    print(response['messages'][-1].content)
    
    
if __name__ == '__main__':
    asyncio.run(main())
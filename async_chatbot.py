from typing import TypedDict, Annotated, List

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
import sqlite3
from langchain.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchResults
import aiosqlite
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

    
llm = ChatOpenAI(model="gpt-4o-mini")


client = MultiServerMCPClient(
    {
        "ExpenseTracker": {
            "transport": "streamable_http",
            "url": "http://127.0.0.1:8000/mcp"
        }
    }
)






class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

async def build_graph():
    
    tools = await client.get_tools()
    
    llm_with_tools = llm.bind_tools(tools)
    
    async_conn = await aiosqlite.connect("aio_chatbot.db")
    
    # Async checkpointer
    checkpointer = AsyncSqliteSaver(conn= async_conn)

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


    
    

    chatbot = graph.compile(checkpointer=checkpointer)
    
    return chatbot , checkpointer


async def retrieve_all_threads(checkpointer: AsyncSqliteSaver):
    threads = []
    async for cp in checkpointer.alist(None):
        tid = cp.config["configurable"]["thread_id"]
        if tid not in threads:
            threads.append(tid)
    return threads[::-1]
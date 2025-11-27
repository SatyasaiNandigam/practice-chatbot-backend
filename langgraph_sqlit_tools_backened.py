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

load_dotenv()

class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    
llm = ChatOpenAI(model="gpt-4o-mini")

search_tool = DuckDuckGoSearchResults()

@tool
def calculator(a:int, b: int, operation: str):
    """
    Perform a basic operation on two numbers
    Supported operations: add, sub, mul, div
    """
    
    try:
        if operation == "add":
            return a + b
        if operation == "sub":
            return a - b
        if operation == "mul":
            return a * b
        if operation == "div":
            if b == 0:
                return {"error": "Can not divide number by Zero"}
            else:
                return a/b
    except Exception as e:
        return {"error": str(e)}
    
tools = [calculator, search_tool]

llm_with_tools = llm.bind_tools(tools)

def chat_node(state: ChatState) -> ChatState:
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)

checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", ToolNode(tools))
graph.set_entry_point("chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")



chatbot = graph.compile(checkpointer= checkpointer)

def retrieve_all_threads():
    threads = []
    for cp in checkpointer.list(None):   # already in creation order
        tid = cp.config["configurable"]["thread_id"]
        if tid not in threads:
            threads.append(tid)
    return threads[::-1]


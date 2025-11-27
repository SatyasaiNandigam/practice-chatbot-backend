from typing import TypedDict, Annotated, List

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    
llm = ChatOpenAI(model="gpt-4o-mini")

def chat_node(state: ChatState) -> ChatState:
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

checkpointer = InMemorySaver()

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.set_entry_point("chat_node")
graph.set_finish_point("chat_node")

chatbot = graph.compile(checkpointer= checkpointer)


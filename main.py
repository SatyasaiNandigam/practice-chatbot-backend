import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

# Your async graph builder + functions
from async_chatbot import build_graph, retrieve_all_threads
from langchain_core.messages import HumanMessage
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import BaseMessage
import json


# ---------------------------- FastAPI App ----------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allows all headers
)


# Global graph + checkpointer
chatbot = None
checkpointer = None


@app.on_event("startup")
async def startup_event():
    """Initialize async graph & checkpointer on startup."""
    global chatbot, checkpointer
    chatbot, checkpointer = await build_graph()
    print("Graph initialized successfully!")


# def serialize_message(msg):
#     """
#     Convert AIMessage / HumanMessage / ToolMessage into a JSON-safe dict.
#     Works even when the message class does NOT implement to_dict().
#     """
#     base = {
#         "type": getattr(msg, "type", None),
#         "content": getattr(msg, "content", None),
#         "id": getattr(msg, "id", None),
#         "additional_kwargs": getattr(msg, "additional_kwargs", {}),
#     }

#     # Tool calls (LLM requested a tool)
#     if hasattr(msg, "tool_calls"):
#         base["tool_calls"] = msg.tool_calls

#     # Response metadata (usage, model details)
#     if hasattr(msg, "response_metadata"):
#         base["response_metadata"] = msg.response_metadata

#     # Usage metadata
#     if hasattr(msg, "usage_metadata"):
#         base["usage_metadata"] = msg.usage_metadata

#     # Name (ToolMessage has 'name')
#     if hasattr(msg, "name"):
#         base["name"] = msg.name

#     return base


def serialize(stream_item):
    """
    stream_item = (chunk, metadata)
    chunk = AIMessageChunk | ToolMessage | AIMessage | ToolMessageChunk
    """
    if not isinstance(stream_item, tuple) or len(stream_item) != 2:
        return None

    chunk, _meta = stream_item
    return serialize_message(chunk)


def serialize_message(msg):
    msg_type = msg.__class__.__name__

    # Determine simplified role
    if msg_type in ("AIMessage", "AIMessageChunk"):
        role = "ai"
    elif msg_type in ("ToolMessage",):
        role = "tool"
    elif msg_type in ("HumanMessage"):
        role = "human"
    else:
        role = "unknown"

    data = {
        "role": role,
        "content": getattr(msg, "content", None) or ""
    }

    # Include tool calls only if present
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        data["tool_calls"] = msg.tool_calls

    if hasattr(msg, "tool_call_chunks") and msg.tool_call_chunks:
        data["tool_call_chunks"] = msg.tool_call_chunks

    return data



class ChatRequest(BaseModel):
    thread_id: str
    message: str
    
    
async def load_conversations(thread_id: str):
    state = await chatbot.aget_state(
        config={"configurable": {"thread_id": thread_id}}
    )

    if state and state.values and "messages" in state.values:
        return state.values["messages"]

    return []







# ---------------------------- API ROUTES ----------------------------

@app.post("/chat")
async def chat_endpoint(payload: ChatRequest):

    async def event_stream():
        async for chunk in chatbot.astream(
            {"messages": [HumanMessage(content=payload.message)]},
            config={"configurable": {"thread_id": payload.thread_id}},
            stream_mode= 'messages'
        ):
           
            # yield f"data: {json.dumps(safe, ensure_ascii=False)}\n\n"
            yield f"data: {(json.dumps(serialize(chunk)))}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/threads")
async def threads_endpoint():
    """
    Fetch all thread IDs from async SQLite checkpointer.
    """
    threads = await retrieve_all_threads(checkpointer)
    return {"threads": threads}


@app.get("/conversations/{thread_id}")
async def get_conversation(thread_id: str):
    try:
        messages = await load_conversations(thread_id)

        serialized = [serialize_message(m) for m in messages]

        return {
            "thread_id": thread_id,
            "messages": serialized
        }

    except Exception as e:
        return {"error": str(e)}


# ---------------------------- Start Server ----------------------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)

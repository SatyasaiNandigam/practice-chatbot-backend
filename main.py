import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

# Your async graph builder + functions
from async_chatbot import build_graph, retrieve_all_threads
from langchain_core.messages import HumanMessage
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware


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


def serialize_message(msg):
    base = {
        "role": getattr(msg, "type", msg.__class__.__name__),
        "content": msg.content
    }

    # 🔹 AIMessage may have multiple tool calls
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        base["tool_calls"] = msg.tool_calls

    # 🔹 ToolMessage contains the id of the tool call
    if hasattr(msg, "tool_call_id") and msg.tool_call_id:
        base["tool_call_id"] = msg.tool_call_id

    return base




# ---------------------------- API ROUTES ----------------------------

@app.post("/chat")
async def chat_endpoint(payload: ChatRequest):
    """
    Stream responses from LangGraph using astream().
    """

    async def event_stream():
        async for chunk in chatbot.astream(
            {"messages": [HumanMessage(content=payload.message)]},
            config={"configurable": {"thread_id": payload.thread_id}},
            stream_mode="messages"
        ):
            # Each 'chunk' is a dict depending on the event
            # You should stream it as text
            yield f"data: {chunk}\n\n"

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
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

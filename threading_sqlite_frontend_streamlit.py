import streamlit as st
from langgraph_sqlit_tools_backened import chatbot, retrieve_all_threads
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid


# ---------------------------------------------- UTility functions -----------------------------------------
def generate_thread():
    thread_id = uuid.uuid4()
    return thread_id

def reset_chat():
    thread_id = generate_thread()
    st.session_state['thread_id'] = thread_id
    add_thread(thread_id)
    st.session_state['message_history'] = []
    
def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)
        
def load_conversations(thread_id):
    if chatbot.get_state(config = {'configurable': {'thread_id': thread_id}}).values:
        return chatbot.get_state(config = {'configurable': {'thread_id': thread_id}}).values['messages']
    
    


# ----------------------------------------------- session setup ------------------------------------------
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

    
if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread()
    
if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()
    
add_thread(st.session_state['thread_id'])
    
#  --------------------------------------------- sidebar ------------------------------------------------
st.sidebar.title("AI Chatbot Langgraph")

if st.sidebar.button("New Chat"):
    reset_chat()
    
    
st.sidebar.header("Conversations")

st.markdown("""
    <style>
    div.stButton > button:active {
        background-color: green !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

for thread_id in reversed(st.session_state['chat_threads']):
    if st.sidebar.button(str(thread_id)):
        st.session_state['thread_id'] = thread_id
        messages = load_conversations(thread_id)
        temp_messages = []
        if messages:
            for msg in messages:
                # ---- 1. Detect and exclude tool calls ----
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    continue

                # For OpenAI/Anthropic style messages
                if hasattr(msg, "additional_kwargs") and "tool_calls" in msg.additional_kwargs:
                    continue

                # ---- 2. Detect and exclude tool call results ----
                if hasattr(msg, "tool_call_id") and msg.tool_call_id:
                    continue

                if hasattr(msg, "additional_kwargs") and "tool_call_id" in msg.additional_kwargs:
                    continue
                
                if isinstance(msg, HumanMessage):
                    role = 'user'
                else:
                    role = 'assistant'
                    
                temp_messages.append({'role': role, 'content': msg.content})
        
        st.session_state['message_history'] = temp_messages
    
# -------------------------------------------------- MAIN UI ---------------------------------------------
 
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])
        

user_input = st.chat_input("Type here")

if user_input:
    st.session_state['message_history'].append({"role": "user", "content": user_input})
    with st.chat_message('user'):
        st.text(user_input)
    
    CONFIG = {
        'configurable': {'thread_id': st.session_state['thread_id']},
        'metadata': {
            'thread_id': st.session_state['thread_id']
        },
        'run_name': 'chat_turn'
        }
    
    
    
    with st.chat_message("assistant"):
        status_box = {"box": None}

        def ai_stream_only():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages"
            ):

                # ---- Tool Events Rendering ----
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_box['box'] is None:
                        status_box['box'] = st.status(
                            f" Using `{tool_name}` ...", expanded=True
                        )
                    else:
                        status_box['box'].update(
                            label= f" Using `{tool_name}` ...",
                            state='running',
                            expanded=True
                        )

                # ---- Stream Assistant Tokens ----
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content


        ai_message = st.write_stream(ai_stream_only())
        
        if status_box['box'] is not None:
            status_box['box'].update(
                label= f"✅ Tool Finished",
                state='complete',
                expanded=False
            )
    

    st.session_state['message_history'].append({"role": "assistant", "content": ai_message})
    
    
    
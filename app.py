import streamlit as st
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

root_path = Path(__file__).resolve().parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from core.inference import InferenceEngine
import time
import uuid

load_dotenv()

st.set_page_config(
    page_title="Owlflow AI",
    page_icon="🦉",
    layout="centered"
)

# --- GEMINI STYLE LIGHT THEME CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Google Sans', sans-serif;
        background-color: #FFFFFF !important;
        color: #1F1F1F !important;
    }

    /* Remove Streamlit branding & space */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .stApp {
        background-color: #FFFFFF;
    }

    /* Sidebar - Soft & Minimal */
    section[data-testid="stSidebar"] {
        background-color: #F8F9FA !important;
        border-right: 1px solid #E0E0E0;
        width: 280px !important;
    }

    /* Chat Containers */
    .chat-row {
        display: flex;
        margin-bottom: 1.5rem;
        width: 100%;
    }

    .user-row {
        justify-content: flex-end;
    }

    .ai-row {
        justify-content: flex-start;
        display: flex;
        gap: 12px;
    }

    .message-card {
        max-width: 80%;
        padding: 1rem 1.25rem;
        font-size: 1rem;
        line-height: 1.5;
        border-radius: 1.5rem;
    }

    .user-card {
        background: #F0F4F8;
        color: #1F1F1F;
        border-radius: 1.5rem 1.5rem 0.25rem 1.5rem;
    }

    .ai-card {
        background: transparent;
        color: #1F1F1F;
    }

    .ai-icon {
        width: 32px;
        height: 32px;
        background: linear-gradient(135deg, #1A73E8 0%, #4285F4 100%);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 14px;
        flex-shrink: 0;
    }

    /* Input Bar - Gemini Style */
    .stChatInputContainer {
        padding-bottom: 2rem !important;
        background: transparent !important;
    }
    
    .stChatInputContainer > div {
        background-color: #F0F4F9 !important;
        border: none !important;
        border-radius: 2rem !important;
        padding-left: 1rem !important;
    }

    /* Headings */
    .main-title {
        font-size: 2.5rem;
        font-weight: 500;
        background: linear-gradient(90deg, #4285F4, #9B72CB, #D96570, #F4AF40);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-top: 3rem;
        margin-bottom: 2rem;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #E0E0E0; border-radius: 10px; }
    ::-webkit-scrollbar-thumb:hover { background: #BDBDBD; }

    </style>
""", unsafe_allow_html=True)

from core.retriever import BusinessRetriever
@st.cache_resource
def get_retriever():
    return BusinessRetriever()

# --- APP LOGIC ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Load the heavy models once
retriever = get_retriever()

if "messages" not in st.session_state:
    try:
        engine = InferenceEngine(st.session_state.session_id, retriever=retriever)
        st.session_state.messages = engine.memory.get_history()
    except:
        st.session_state.messages = []

with st.sidebar:
    st.markdown("<h3 style='color:#444746; margin: 10px 0 30px 0; font-weight:500;'>Owlflow AI</h3>", unsafe_allow_html=True)
    
    # Session management is now hidden from the user but tracked in the backend
    if st.button("+ New Chat", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()
    st.markdown("<div style='position: fixed; bottom: 20px; font-size: 12px; color: #70757A;'>Powered by Groq Llama 3.3</div>", unsafe_allow_html=True)

# Main Title
st.markdown("<div class='main-title'>Hello! How can I help you today?</div>", unsafe_allow_html=True)

# Display Messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
            <div class="chat-row user-row">
                <div class="message-card user-card">{msg["content"]}</div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="chat-row ai-row">
                <div class="ai-icon">🦉</div>
                <div class="message-card ai-card">{msg["content"]}</div>
            </div>
        """, unsafe_allow_html=True)

# User Input
if prompt := st.chat_input("Enter a prompt here"):
    # (No auto-naming needed anymore as we use persistent hidden UUIDs)

    # UI Update for user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f"""
        <div class="chat-row user-row">
            <div class="message-card user-card">{prompt}</div>
        </div>
    """, unsafe_allow_html=True)
    
    with st.spinner(""):
        try:
            # Initialize engine with the cached retriever
            engine = InferenceEngine(st.session_state.session_id, retriever=retriever)
            response, metrics = engine.answer_question(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            st.markdown(f"""
                <div class="chat-row ai-row">
                    <div class="ai-icon">🦉</div>
                    <div class="message-card ai-card">{response}</div>
                </div>
            """, unsafe_allow_html=True)
            
            # Display Latency Metrics
            st.caption(f"⚡ Latency: {metrics['total_time']:.2f}s (Retrieval: {metrics['retrieval_time']:.2f}s | Inference: {metrics['inference_time']:.2f}s)")
            
        except Exception as e:
            st.error(f"Error: {e}")

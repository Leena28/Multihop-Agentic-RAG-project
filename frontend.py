import streamlit as st
import requests
import os

# Page Configuration
st.set_page_config(page_title="NEXUS RAG", page_icon="◈", layout="wide")

# Styling with CSS
st.markdown("""<style>.stApp {background-color: #0a0a0a; color: white;}
    section[data-testid="stSidebar"] {background-color: #111111;border-right: 1px solid #2b2b2b;}.main-title {
        font-size: 3rem;
        font-weight: 800;
        color: #facc15;
        margin-bottom: 0;
        letter-spacing: 1px;
    }
    .sub-title {
        color: #bbbbbb;
        margin-top: -10px;
        margin-bottom: 30px;
        font-size: 1rem;
    }
    .stChatMessage {
        border-radius: 16px;
        padding: 12px;
        margin-bottom: 12px;
        border: 1px solid #222;
        background-color: #151515;
    }
    [data-testid="stChatMessageContent"] {
        font-size: 16px;
        line-height: 1.7;
    }
    .stButton > button {
        background-color: #facc15;
        color: black;
        border: none;
        border-radius: 12px;
        font-weight: 700;
        width: 100%;
        transition: 0.3s;
    }
    .stButton > button:hover {
        background-color: #eab308;
        color: black;
    }
    .stChatInput textarea {
        background-color: #151515 !important;
        color: white !important;
        border-radius: 14px !important;
        border: 1px solid #333 !important;
    }
    .status-card {
        background-color: #151515;
        padding: 16px;
        border-radius: 16px;
        border: 1px solid #2b2b2b;
        margin-bottom: 16px;
    }
    .status-title {
        color: #facc15;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .healthy {color: #22c55e; font-weight: bold;}
    .unhealthy {color: #ef4444; font-weight: bold;}
    .source-box {
        background-color: #121212;
        border-left: 4px solid #facc15;
        padding: 12px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .meta-box {
        background-color: #151515;
        border: 1px solid #2a2a2a;
        border-radius: 12px;
        padding: 12px 16px;
        margin-top: 12px;
    }
    .meta-row {
        display: flex;
        justify-content: space-between;
        margin-bottom: 6px;
    }
    .meta-label {color: #aaaaaa; font-size: 13px;}
    .meta-value {color: #facc15; font-size: 13px; font-weight: 700;}
    .pass {color: #22c55e; font-weight: bold;}
    .fail {color: #ef4444; font-weight: bold;}
    </style>
""", unsafe_allow_html=True)

# Sesion State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "backend_ok" not in st.session_state:
    st.session_state.backend_ok = False

# Backend url
#BASE_URL = "http://localhost:8000"
BASE_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# BACKEND HEALTH CHECK
def check_backend():
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

# get response from api
def get_response(prompt):
    response = requests.post(f"{BASE_URL}/query",json={"question": prompt})
    return response.json()

# Header
st.markdown("""<div class="main-title">◈ NEXUS RAG</div>
<div class="sub-title">Agentic Multi-Hop Retrieval Intelligence System</div>""", unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    st.markdown("## ◉ Control Center")

    if st.button("Check Backend Status"):
        st.session_state.backend_ok = check_backend()

    if st.session_state.backend_ok:
        st.markdown("""
            <div class="status-card">
                <div class="status-title">Backend Status</div>
                <div class="healthy">ONLINE</div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="status-card">
                <div class="status-title">Backend Status</div>
                <div class="unhealthy">OFFLINE</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

# Display Chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and "metadata" in message:
            meta = message["metadata"]

            grounded_class = "pass" if meta.get("is_grounded") else "fail"
            grounded_text = "YES" if meta.get("is_grounded") else "NO"
            addresses_class = "pass" if meta.get("addresses_question") else "fail"
            addresses_text = "YES" if meta.get("addresses_question") else "NO"

            st.markdown(f"""
                <div class="meta-box">
                    <div class="meta-row">
                        <span class="meta-label">Grounded</span>
                        <span class="{grounded_class}">{grounded_text}</span>
                    </div>
                    <div class="meta-row">
                        <span class="meta-label">Addresses Question</span>
                        <span class="{addresses_class}">{addresses_text}</span>
                    </div>
                    <div class="meta-row">
                        <span class="meta-label">Attempts</span>
                        <span class="meta-value">{meta.get("attempt_count", 0)}</span>
                    </div>
                    <div class="meta-row">
                        <span class="meta-label">Latency</span>
                        <span class="meta-value">{meta.get("latency_seconds", 0)}s</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

prompt = st.chat_input("Ask your AI research question...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        try:
            with st.spinner("Running Agentic RAG Pipeline..."):
                data = get_response(prompt)

            answer = data.get("answer", "No answer returned")
            message_placeholder.markdown(answer)

            metadata = {
                "is_grounded": data.get("is_grounded", False),
                "addresses_question": data.get("addresses_question", False),
                "attempt_count": data.get("attempt_count", 0),
                "latency_seconds": data.get("latency_seconds", 0),
                "sources": data.get("sources", [])
            }

            grounded_class = "pass" if metadata["is_grounded"] else "fail"
            grounded_text = "YES" if metadata["is_grounded"] else "NO"
            addresses_class = "pass" if metadata["addresses_question"] else "fail"
            addresses_text = "YES" if metadata["addresses_question"] else "NO"

            st.markdown(f"""
                <div class="meta-box">
                    <div class="meta-row">
                        <span class="meta-label">Grounded</span>
                        <span class="{grounded_class}">{grounded_text}</span>
                    </div>
                    <div class="meta-row">
                        <span class="meta-label">Addresses Question</span>
                        <span class="{addresses_class}">{addresses_text}</span>
                    </div>
                    <div class="meta-row">
                        <span class="meta-label">Attempts</span>
                        <span class="meta-value">{metadata["attempt_count"]}</span>
                    </div>
                    <div class="meta-row">
                        <span class="meta-label">Latency</span>
                        <span class="meta-value">{metadata["latency_seconds"]}s</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            answer = f"System Error: {str(e)}"
            message_placeholder.error(answer)
            metadata = {}

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "metadata": metadata
    })
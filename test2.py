import streamlit as st
import pandas as pd
import uuid
import msgpack
from msgpack import ExtType
from pymongo import MongoClient
import os
import time
import json
from datetime import datetime
from typing import List, Dict, Optional
import plotly.graph_objects as go
from io import StringIO

# Import your existing modules
from src.app.entry import run_app
from src.utils.openai_api import get_supervisor_llm
from config.config import SHIPMENT_DF_PATH, RATECARD_PATH, INSIGHTS_DATA_PATH, SKU_MASTER_PATH
from src.utils.blob_access import AzureBlobStorage
from config.config import AZURE_CONNECTION_STRING, AZURE_CONTAINER_NAME, AZURE_BLOB_URL_BASE

# MongoDB Configuration
MONGODB_URI = "mongodb+srv://copilotgenai:CFMEEGr0T6O63wuk@clusterai.nd3uoy.mongodb.net/?retryWrites=true&w=majority&appName=Clusterai"
DB_NAME = "checkpointing_db"

# Page Configuration
st.set_page_config(
    page_title="Logistics Analytics Assistant",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS with beautiful styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global Styles */
    .main {
        font-family: 'Inter', sans-serif;
        padding-bottom: 140px;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        min-height: 100vh;
    }

    /* Hide default Streamlit elements */
    .stChatMessage {
        display: none !important;
    }

    /* Enhanced Header Styles with glassmorphism effect */
    .chat-header {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.9) 0%, rgba(118, 75, 162, 0.9) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 2rem;
        border-radius: 20px 20px 20px 20px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
        text-align: center;
        position: relative;
        overflow: hidden;
    }

    .chat-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="25" cy="25" r="1" fill="rgba(255,255,255,0.1)"/><circle cx="75" cy="75" r="1" fill="rgba(255,255,255,0.1)"/><circle cx="50" cy="10" r="0.5" fill="rgba(255,255,255,0.05)"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
        opacity: 0.3;
        pointer-events: none;
    }

    .chat-header h1 {
        color: white;
        margin: 0;
        font-weight: 800;
        font-size: 2.2rem;
        text-shadow: 0 2px 10px rgba(0,0,0,0.3);
        position: relative;
        z-index: 1;
    }

    .chat-header p {
        color: rgba(255,255,255,0.95);
        margin: 0.5rem 0 0 0;
        font-size: 1rem;
        font-weight: 400;
        position: relative;
        z-index: 1;
    }

    /* Enhanced Sidebar Styles */
    .sidebar-header {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 8px 25px rgba(79, 172, 254, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    .sidebar-header h3 {
        color: white;
        margin: 0;
        font-weight: 700;
        font-size: 1.3rem;
        text-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }

    /* Enhanced Chat Container - Connected to header with perfect alignment */
    .chat-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 0 0 20px 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
        min-height: 500px;
        max-height: 600px;
        overflow-y: auto;
        border-top: none;
        position: relative;
    }

    /* Enhanced Message Styles */
    .message-container {
        display: flex;
        margin-bottom: 1.5rem;
        width: 100%;
        animation: fadeInUp 0.5s ease-out;
    }

    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .user-message-container {
        justify-content: flex-end;
    }

    .bot-message-container {
        justify-content: flex-start;
    }

    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-size: 1.2rem;
        color: white;
        padding: 1.2rem 1.8rem;
        border-radius: 25px 25px 8px 25px;
        max-width: 75%;
        word-wrap: break-word;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        margin-left: auto;
        position: relative;
        font-weight: 500;
    }

    .user-message::before {
        content: '';
        position: absolute;
        bottom: -8px;
        right: 15px;
        width: 0;
        height: 0;
        border-left: 8px solid transparent;
        border-right: 8px solid transparent;
        border-top: 8px solid #764ba2;
    }

    .bot-message {
        background: rgba(248, 249, 250, 0.9);
        font-size: 1.2rem;
        backdrop-filter: blur(10px);
        color: #2d3748;
        padding: 1.2rem 1.8rem;
        border-radius: 25px 25px 25px 8px;
        max-width: 75%;
        word-wrap: break-word;
        border-left: 4px solid #667eea;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        margin-right: auto;
        position: relative;
        font-weight: 500;
    }

    .bot-message::before {
        content: '';
        position: absolute;
        bottom: -8px;
        left: 15px;
        width: 0;
        height: 0;
        border-left: 8px solid transparent;
        border-right: 8px solid transparent;
        border-top: 8px solid rgba(248, 249, 250, 0.9);
    }

    /* Enhanced Chart Container */
    .chart-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        border: 1px solid rgba(255, 255, 255, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .chart-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 35px rgba(0,0,0,0.15);
    }

    /* Enhanced Input Area - Fixed at bottom with perfect alignment */
    .input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        padding: 2rem;
        border-radius: 20px 20px 0 0;
        box-shadow: 0 -8px 32px rgba(0,0,0,0.15);
        z-index: 1000;
        border-top: 1px solid rgba(255, 255, 255, 0.3);
    }

    /* Perfect alignment for input container content */
    .input-container-content {
        max-width: 1200px;
        margin: 0 auto;
        padding-left: 320px; /* Match sidebar width exactly */
        padding-right: 1rem;
    }

    /* Enhanced Dashboard Modal */
    .dashboard-modal {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0,0,0,0.85);
        backdrop-filter: blur(10px);
        z-index: 9999;
        display: flex;
        justify-content: center;
        align-items: center;
        animation: fadeIn 0.3s ease-out;
    }

    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    .dashboard-content {
        background: rgba(255, 255, 255, 0.98);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 2.5rem;
        max-width: 95%;
        max-height: 90%;
        overflow-y: auto;
        position: relative;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        border: 1px solid rgba(255, 255, 255, 0.3);
        animation: slideIn 0.4s ease-out;
    }

    @keyframes slideIn {
        from {
            opacity: 0;
            transform: scale(0.9) translateY(20px);
        }
        to {
            opacity: 1;
            transform: scale(1) translateY(0);
        }
    }

    .close-button {
        position: absolute;
        top: 1.5rem;
        right: 1.5rem;
        background: linear-gradient(135deg, #ff4757 0%, #ff3742 100%);
        color: white;
        border: none;
        border-radius: 50%;
        width: 45px;
        height: 45px;
        cursor: pointer;
        font-size: 1.3rem;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 71, 87, 0.4);
    }

    .close-button:hover {
        transform: scale(1.1);
        box-shadow: 0 6px 20px rgba(255, 71, 87, 0.6);
    }

    /* Enhanced Follow-up Questions */
    .followup-container {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 1rem;
        border-radius: 20px;
        margin: 2rem 0;
        box-shadow: 0 8px 25px rgba(250, 112, 154, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    .followup-container h4 {
        color: white;
        margin: 0 0 1.5rem 0;
        font-weight: 700;
        font-size: 1.2rem;
        text-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }

    /* Enhanced Status Indicators */
    .status-indicator {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.3rem;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
    }

    .status-online {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        color: #155724;
        border: 1px solid rgba(21, 87, 36, 0.2);
    }

    .status-processing {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        color: #856404;
        border: 1px solid rgba(133, 100, 4, 0.2);
    }

    /* Enhanced Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 30px;
        padding: 0.7rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        font-family: 'Inter', sans-serif;
    }

    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
    }

    /* Enhanced Metrics Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.8rem;
        border: 1px solid rgba(255, 255, 255, 0.3);
        transition: transform 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-5px);
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.8rem 0;
    }

    .metric-label {
        color: #4a5568;
        font-size: 1rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Enhanced Loading Animation */
    .typing-indicator {
        display: flex;
        align-items: center;
        padding: 1.2rem;
        background: rgba(248, 249, 250, 0.9);
        backdrop-filter: blur(10px);
        border-radius: 25px;
        margin: 0.8rem auto 0.8rem 0;
        max-width: 100px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }

    .typing-dot {
        width: 10px;
        height: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 50%;
        margin: 0 3px;
        animation: typing 1.6s infinite ease-in-out;
    }

    .typing-dot:nth-child(1) { animation-delay: -0.32s; }
    .typing-dot:nth-child(2) { animation-delay: -0.16s; }

    @keyframes typing {
        0%, 80%, 100% { 
            transform: scale(0.8); 
            opacity: 0.5; 
        }
        40% { 
            transform: scale(1.2); 
            opacity: 1; 
        }
    }

    /* Enhanced Scrollbar Styling */
    .chat-container::-webkit-scrollbar {
        width: 8px;
    }

    .chat-container::-webkit-scrollbar-track {
        background: rgba(241, 241, 241, 0.5);
        border-radius: 10px;
    }

    .chat-container::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }

    .chat-container::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
    }

    /* Enhanced Welcome Message */
    .welcome-message {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.95) 0%, rgba(118, 75, 162, 0.95) 100%);
        backdrop-filter: blur(20px);
        padding: 2.5rem;
        border-radius: 20px;
        margin: 2rem 0;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    .welcome-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }

    .welcome-card {
        background: rgba(255,255,255,0.15);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: transform 0.3s ease;
    }

    .welcome-card:hover {
        transform: translateY(-5px);
        background: rgba(255,255,255,0.2);
    }

    /* Dashboard Grid Layout */
    .dashboard-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
        gap: 2rem;
        margin: 2rem 0;
    }

    .dashboard-chart-card {
        background: rgba(255, 255, 255, 0.98);
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        border: 1px solid rgba(255, 255, 255, 0.3);
        transition: transform 0.3s ease;
    }

    .dashboard-chart-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 35px rgba(0,0,0,0.15);
    }

    /* Responsive Design */
    @media (max-width: 768px) {
        .user-message, .bot-message {
            max-width: 95%;
        }

        .chat-header h1 {
            font-size: 1.8rem;
        }

        .input-container-content {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .main {
            padding-bottom: 160px;
        }

        .dashboard-grid {
            grid-template-columns: 1fr;
        }
    }

    /* Form styling */
    .stTextArea > div > div > textarea {
        border-radius: 15px;
        border: 2px solid rgba(102, 126, 234, 0.2);
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
    }

    .stTextArea > div > div > textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
</style>
""", unsafe_allow_html=True)


def get_conversation(user_id: str) -> Dict:
    """Retrieve conversation history for a user from MongoDB"""
    try:
        chat_history = {}
        client = MongoClient(MONGODB_URI)
        db = client[DB_NAME]
        collection = db["checkpoints"]
        query = {"metadata.user_id": bytes(f'"{user_id}"', 'utf-8')}
        filtered_docs = list(collection.find(query))

        for doc in filtered_docs:
            thread_id = doc['thread_id']
            if thread_id not in chat_history:
                chat_history[thread_id] = []
            else:
                raw_data = doc['checkpoint']
                unpacked_raw_data = msgpack.unpackb(raw_data, raw=False)
                channel_data = unpacked_raw_data['channel_values']
                ext_messages = channel_data['messages']

                decoded_messages = []
                for ext in ext_messages:
                    if isinstance(ext, ExtType):
                        decoded = msgpack.unpackb(ext.data, raw=False)
                        decoded_messages.append(decoded)

                for msg in decoded_messages:
                    if msg[1] == 'HumanMessage':
                        role = "user"
                    else:
                        role = "bot"
                    content = msg[2]['content']
                    chat_history[thread_id].append({
                        "role": role,
                        "content": content,
                        "img_list": [],
                        "timestamp": datetime.now().isoformat()
                    })

        client.close()
        return chat_history
    except Exception as e:
        st.error(f"Error retrieving conversations: {str(e)}")
        return {}


def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
        "user_id": None,
        "thread_id": uuid.uuid4().hex,
        "selected_thread": None,
        "next_question": None,
        "chat_history": [],
        "follow_up": [],
        "image": [],
        "all_plots": [],  # New session state for storing all plots
        "feedback": None,
        "chat_input_text": "",
        "is_processing": False,
        "conversation_count": 0,
        "user_chat_history": {},
        "show_welcome": True,
        "show_dashboard": False,
        "all_thread_images": [],
        "theme": "light"
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_data
def load_data():
    """Load and cache data files"""
    try:
        data = {
            "shipment_df": pd.read_excel(SHIPMENT_DF_PATH, sheet_name="Sheet1"),
            "rate_card": pd.read_excel(RATECARD_PATH),
            "insights_df": pd.read_csv(INSIGHTS_DATA_PATH),
            "SKU_master": pd.read_csv(SKU_MASTER_PATH)
        }
        return data
    except Exception as e:
        st.error(f"Error loading data files: {str(e)}")
        return None


@st.cache_resource
def initialize_services():
    """Initialize Azure client and LLM"""
    try:
        azure_client = AzureBlobStorage(
            connection_string=AZURE_CONNECTION_STRING,
            container_name=AZURE_CONTAINER_NAME,
            blob_url_base=AZURE_BLOB_URL_BASE
        )
        # Load secrets into environment variables
        for key, value in st.secrets.items():
            os.environ[key] = value
        api_key = os.getenv("OPENAI_API_KEY")
        llm = get_supervisor_llm(api_key)

        return azure_client, llm
    except Exception as e:
        st.error(f"Error initializing services: {str(e)}")
        return None, None


def render_sidebar():
    """Render enhanced sidebar with user management and conversation history"""
    with st.sidebar:
        # Enhanced Sidebar Header
        st.markdown("""
        <div class="sidebar-header">
            <h3>🎯 Control Panel</h3>
        </div>
        """, unsafe_allow_html=True)

        # User Authentication Section
        with st.expander("👤 User Authentication", expanded=True):
            with st.form(key="user_auth_form", clear_on_submit=False):
                user_id = st.text_input(
                    "Enter User ID",
                    placeholder="Enter your unique identifier",
                    help="This helps us maintain your conversation history",
                    value=st.session_state.user_id
                )

                col1, col2 = st.columns(2)
                with col1:
                    login_btn = st.form_submit_button("🔐 Login", use_container_width=True)
                with col2:
                    logout_btn = st.form_submit_button("🚪 Logout", use_container_width=True)

                if login_btn and user_id:
                    if user_id != st.session_state.user_id:
                        st.session_state.user_id = user_id
                        st.session_state.user_chat_history = get_conversation(user_id)
                        st.session_state.conversation_count = len(st.session_state.user_chat_history)
                        st.success(f"✅ Logged in as: {user_id}")
                        st.rerun()

                if logout_btn:
                    for key in ["user_id", "user_chat_history", "chat_history", "selected_thread", "all_plots"]:
                        if key in st.session_state:
                            st.session_state[key] = None if key == "user_id" else ([] if key == "all_plots" else {})
                    st.success("👋 Logged out successfully")
                    st.rerun()

        # User Status Display
        if st.session_state.user_id:
            st.markdown(f"""
            <div class="status-indicator status-online">
                🟢 Connected: {st.session_state.user_id}
            </div>
            """, unsafe_allow_html=True)

            # Enhanced Dashboard Button
            if st.button("📊 Show Dashboard", use_container_width=True, key="dashboard_btn"):
                st.session_state.show_dashboard = True
                st.rerun()

            # Enhanced Conversation Statistics
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{st.session_state.conversation_count}</div>
                    <div class="metric-label">Conversations</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{len(st.session_state.chat_history)}</div>
                    <div class="metric-label">Messages</div>
                </div>
                """, unsafe_allow_html=True)

            # Plots Statistics
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(st.session_state.all_plots)}</div>
                <div class="metric-label">Generated Charts</div>
            </div>
            """, unsafe_allow_html=True)

            # Conversation History
            st.markdown("### 💬 Conversation History")

            # Search conversations
            search_term = st.text_input("🔍 Search conversations", placeholder="Search by keywords...")

            chat_history = st.session_state.user_chat_history
            filtered_conversations = {}

            if search_term:
                for thread_id, messages in chat_history.items():
                    for message in messages:
                        if search_term.lower() in message["content"].lower():
                            filtered_conversations[thread_id] = messages
                            break
            else:
                filtered_conversations = chat_history

            # New Conversation Button
            if st.button("➕ Start New Conversation", use_container_width=True):
                st.session_state.selected_thread = "New_Conversation"
                st.rerun()

            # Display conversations
            if filtered_conversations:
                for thread_id, messages in list(filtered_conversations.items())[:10]:
                    if messages:
                        preview = messages[0]["content"][:50] + "..."
                        timestamp = messages[0].get("timestamp", "Unknown time")

                        if st.button(
                                f"💭 {preview}",
                                key=f"conv_{thread_id}",
                                help=f"Started: {timestamp}",
                                use_container_width=True
                        ):
                            st.session_state.selected_thread = thread_id
                            st.rerun()
            else:
                st.info("No conversations found matching your search.")

        # Export Options
        if st.session_state.chat_history:
            st.markdown("### 📥 Export Options")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("📄 Export JSON", use_container_width=True):
                    export_data = {
                        "user_id": st.session_state.user_id,
                        "thread_id": st.session_state.thread_id,
                        "conversation": st.session_state.chat_history,
                        "plots_count": len(st.session_state.all_plots),
                        "exported_at": datetime.now().isoformat()
                    }
                    st.download_button(
                        "⬇️ Download",
                        data=json.dumps(export_data, indent=2),
                        file_name=f"conversation_{st.session_state.thread_id[:8]}.json",
                        mime="application/json"
                    )

            with col2:
                if st.button("📊 Export CSV", use_container_width=True):
                    df = pd.DataFrame(st.session_state.chat_history)
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "⬇️ Download",
                        data=csv,
                        file_name=f"conversation_{st.session_state.thread_id[:8]}.csv",
                        mime="text/csv"
                    )


def get_result(question: str, data: Dict, azure_client, llm):
    """Get response from the AI model"""
    try:
        # Reset image list before getting new result
        st.session_state.image = []

        result = run_app(
            llm=llm,
            question=question,
            thread_id=st.session_state.thread_id,
            user_id=st.session_state.user_id,
            shipment_df=data["shipment_df"],
            rate_card=data["rate_card"],
            insights_df=data["insights_df"],
            SKU_master=data["SKU_master"],
            azure_client=azure_client
        )

        # Store plots in all_plots session state
        if st.session_state.image:
            st.session_state.all_plots.extend(st.session_state.image)

        return result
    except Exception as e:
        st.error(f"Error getting AI response: {str(e)}")
        return {"messages": [{"agent": "Error", "text": f"Sorry, I encountered an error: {str(e)}"}], "follow_up": []}


def prepare_final_text(result: Dict) -> List[str]:
    """Format the AI response text"""
    response_text_list = []
    for msg in result.get('messages', []):
        if msg:
            agent = msg.get('agent', 'AI')
            text = msg.get('text', '')
            response_text_list.append(f"**{agent}**: {text}")
    return response_text_list


def render_typing_indicator():
    """Show enhanced typing indicator while processing"""
    st.markdown("""
    <div class="typing-indicator">
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
    </div>
    """, unsafe_allow_html=True)


def render_welcome_message():
    """Render enhanced welcome message for new users"""
    if st.session_state.show_welcome and not st.session_state.chat_history:
        welcome_html = """
        <div class="welcome-message">
            <h2>👋 Welcome to Logistics Analytics Assistant!</h2>
            <p>I'm here to help you with comprehensive logistics insights and analytics</p>
            <div class="welcome-grid">
                <div class="welcome-card">
                    <strong>📊 Data Analysis</strong><br>
                    Shipment insights and trends
                </div>
                <div class="welcome-card">
                    <strong>💰 Cost Optimization</strong><br>
                    Transport cost analysis
                </div>
                <div class="welcome-card">
                    <strong>📈 Performance Metrics</strong><br>
                    KPI tracking and performance analysis
                </div>
                <div class="welcome-card">
                    <strong>🎯 Strategic Insights</strong><br>
                    Business intelligence
                </div>
            </div>
            <p><em>Try asking: "What are our top shipping routes?" or "Show me cost analysis for this month"</em></p>
        </div>
        """
        st.markdown(welcome_html, unsafe_allow_html=True)

        # Sample questions
        sample_questions = [
            'Please tell about your capabilities and datasets',
            'Brief me about key agents and optimisation strategies available',
            'Provide me some examples of Insight Questions',
            'Provide me some examples of Order Frequency Optimization Questions',
            'Provide me some examples of Drop Location Centralization Questions',
            'Provide me some examples of Pallet Utilization Questions'
        ]

        st.markdown("**🚀 Quick Start Questions:**")
        cols = st.columns(2)
        for i, question in enumerate(sample_questions):
            with cols[i % 2]:
                if st.button(question, key=f"sample_{i}", use_container_width=True):
                    st.session_state.chat_input_text = question
                    st.session_state.show_welcome = False
                    st.rerun()


def display_chart(chart, chart_index: int, message_index: int, chart_type: str = "user"):
    """Display charts with enhanced styling and proper handling for both Plotly and Matplotlib"""
    try:

        # Add enhanced chart container styling
        # st.markdown('<div class="chart-container">', unsafe_allow_html=True)

        # Check if it's a Plotly chart (has update_layout method)
        if hasattr(chart, 'update_layout'):
            # It's a Plotly figure
            chart.update_layout(
                width=1000,
                height=500,
                margin=dict(l=20, r=20, t=40, b=20),
                font=dict(family="Inter, sans-serif", size=12),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                title_font_size=16,
                title_font_color='#2d3748'
            )
            # Create a unique key for each chart
            chart_key = f"{chart_type}_chart_{message_index}_{chart_index}_{uuid.uuid4().hex[:8]}"
            st.plotly_chart(chart, use_container_width=True, key=chart_key)

        # Check if it's a Matplotlib figure (has savefig method)
        elif hasattr(chart, 'savefig'):
            # It's a Matplotlib figure
            # Create a unique key for each chart
            st.pyplot(chart, use_container_width=True)

        # Check if it's a Matplotlib Axes object
        elif hasattr(chart, 'figure'):
            # It's a Matplotlib Axes object, get the figure
            st.pyplot(chart.figure, use_container_width=True)

        # Check if it's an image URL or path
        elif isinstance(chart, str):
            # It's likely an image URL or path
            if chart.startswith(('http://', 'https://', 'data:')):
                st.image(chart, caption="Generated Chart", use_container_width=True)
            else:
                # Try to display as file path
                try:
                    st.image(chart, caption="Generated Chart", use_container_width=True)
                except:
                    st.error(f"Could not display image from path: {chart}")

        # Check if it's a PIL Image
        elif hasattr(chart, 'save'):
            # It's likely a PIL Image
            st.image(chart, caption="Generated Chart", use_container_width=True)

        # Check if it's a numpy array (image data)
        elif hasattr(chart, 'shape') and len(chart.shape) in [2, 3]:
            # It's likely a numpy array representing an image
            st.image(chart, caption="Generated Chart", use_container_width=True)

        # Handle other chart types or objects
        else:
            # Try to display as is - might be a custom chart object
            try:
                st.write(chart)
            except Exception as e:
                st.error(f"Unsupported chart type: {type(chart)}. Error: {str(e)}")
                st.write("Chart object details:", {
                    "type": str(type(chart)),
                    "attributes": [attr for attr in dir(chart) if not attr.startswith('_')][:10]
                })

        st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error displaying chart: {str(e)}")
        st.write("Chart details:", {
            "type": str(type(chart)),
            "has_update_layout": hasattr(chart, 'update_layout'),
            "has_savefig": hasattr(chart, 'savefig'),
            "has_figure": hasattr(chart, 'figure'),
            "is_string": isinstance(chart, str)
        })


# def render_dashboard_modal():
#     """Render enhanced dashboard modal with all plots"""
#     if st.session_state.show_dashboard:
#         # Create modal overlay with enhanced styling
#         st.markdown("""
#         <div class="dashboard-modal">
#             <div class="dashboard-content">
#                 <button class="close-button">×</button>
#                 <h2 style="text-align: center; color: #2d3748; margin-bottom: 1rem;">📊 Analytics Dashboard</h2>
#                 <p style="text-align: center; color: #4a5568; margin-bottom: 2rem;">All charts and visualizations from your session</p>
#             </div>
#         </div>
#         """, unsafe_allow_html=True)
#
#         # Close button functionality
#         col1, col2, col3 = st.columns([1, 2, 1])
#         with col2:
#             if st.button("❌ Close Dashboard", key="close_dashboard", use_container_width=True):
#                 st.session_state.show_dashboard = False
#                 st.rerun()
#
#         # Display dashboard statistics
#         col1, col2, col3 = st.columns(3)
#         with col1:
#             st.markdown(f"""
#             <div class="metric-card">
#                 <div class="metric-value">{len(st.session_state.all_plots)}</div>
#                 <div class="metric-label">Total Charts</div>
#             </div>
#             """, unsafe_allow_html=True)
#
#         with col2:
#             st.markdown(f"""
#             <div class="metric-card">
#                 <div class="metric-value">{len(st.session_state.chat_history)}</div>
#                 <div class="metric-label">Messages</div>
#             </div>
#             """, unsafe_allow_html=True)
#
#         with col3:
#             st.markdown(f"""
#             <div class="metric-card">
#                 <div class="metric-value">{st.session_state.thread_id[:8]}</div>
#                 <div class="metric-label">Session ID</div>
#             </div>
#             """, unsafe_allow_html=True)
#
#         # Display all plots in a beautiful grid
#         if st.session_state.all_plots:
#             st.markdown("### 📈 Generated Charts & Visualizations")
#
#             # Create a responsive grid layout
#             st.markdown('<div class="dashboard-grid">', unsafe_allow_html=True)
#
#             # Display charts in grid format
#             cols = st.columns(2)
#             for i, chart in enumerate(st.session_state.all_plots):
#                 with cols[i % 2]:
#                     st.markdown('<div class="dashboard-chart-card">', unsafe_allow_html=True)
#                     st.markdown(f"**Chart {i + 1}**")
#                     display_chart(chart, i, 0, "dashboard")
#                     st.markdown('</div>', unsafe_allow_html=True)
#
#             st.markdown('</div>', unsafe_allow_html=True)
#
#             # Export dashboard option
#             st.markdown("---")
#             col1, col2, col3 = st.columns([1, 2, 1])
#             with col2:
#                 if st.button("📥 Export Dashboard Data", use_container_width=True):
#                     dashboard_data = {
#                         "session_id": st.session_state.thread_id,
#                         "user_id": st.session_state.user_id,
#                         "total_charts": len(st.session_state.all_plots),
#                         "total_messages": len(st.session_state.chat_history),
#                         "exported_at": datetime.now().isoformat(),
#                         "conversation_summary": [msg["content"][:100] + "..." for msg in st.session_state.chat_history
#                                                  if msg["role"] == "user"]
#                     }
#                     st.download_button(
#                         "⬇️ Download Dashboard Summary",
#                         data=json.dumps(dashboard_data, indent=2),
#                         file_name=f"dashboard_{st.session_state.thread_id[:8]}.json",
#                         mime="application/json",
#                         use_container_width=True
#                     )
#         else:
#             st.markdown("""
#             <div style="text-align: center; padding: 3rem; color: #4a5568;">
#                 <h3>📊 No Charts Generated Yet</h3>
#                 <p>Start asking questions to generate beautiful visualizations!</p>
#             </div>
#             """, unsafe_allow_html=True)


def render_dashboard_modal():
    """Render enhanced dashboard modal with all plots"""
    if st.session_state.get('show_dashboard', False):
        # Add custom CSS for styling
        st.markdown("""
        <style>
        .dashboard-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 15px;
            margin: 1rem 0;
            border: 1px solid #e2e8f0;
        }
        .metric-card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            text-align: center;
            margin: 0.5rem 0;
        }
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            color: #2d3748;
            margin-bottom: 0.5rem;
        }
        .metric-label {
            font-size: 0.9rem;
            color: #4a5568;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .dashboard-chart-card {
            background: white;
            padding: 1rem;
            border-radius: 10px;
            margin: 0.5rem 0;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        .dashboard-grid {
            margin: 1rem 0;
        }
        .dashboard-header {
            text-align: center;
            color: white;
            margin-bottom: 1rem;
        }
        .dashboard-header h2 {
            color: white;
            margin: 0;
        }
        .dashboard-header p {
            color: rgba(255, 255, 255, 0.8);
            margin: 0.5rem 0;
        }
        </style>
        """, unsafe_allow_html=True)

        # Dashboard container
        st.markdown("""
        <div class="dashboard-container">
            <div class="dashboard-header">
                <h2>📊 Analytics Dashboard</h2>
                <p>All charts and visualizations from your session</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Close button functionality
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("❌ Close Dashboard", key="close_dashboard", use_container_width=True):
                st.session_state.show_dashboard = False
                st.rerun()

        # Display dashboard statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            total_plots = len(st.session_state.get('all_plots', []))
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{total_plots}</div>
                <div class="metric-label">Total Charts</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            total_messages = len(st.session_state.get('chat_history', []))
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{total_messages}</div>
                <div class="metric-label">Messages</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            session_id = st.session_state.get('thread_id', 'N/A')
            display_id = session_id[:8] if session_id != 'N/A' else 'N/A'
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{display_id}</div>
                <div class="metric-label">Session ID</div>
            </div>
            """, unsafe_allow_html=True)

        # Display all plots in a beautiful grid
        if st.session_state.get('all_plots') and len(st.session_state.all_plots) > 0:
            st.markdown("### 📈 Generated Charts & Visualizations")

            # Create a responsive grid layout
            st.markdown('<div class="dashboard-grid">', unsafe_allow_html=True)

            # Display charts in grid format
            cols = st.columns(2)
            for i, chart in enumerate(st.session_state.all_plots):
                with cols[i % 2]:
                    st.markdown('<div class="dashboard-chart-card">', unsafe_allow_html=True)
                    st.markdown(f"**Chart {i + 1}**")

                    # Display chart based on type
                    try:
                        # if hasattr(chart, 'show'):  # Plotly chart
                        if hasattr(chart, 'to_plotly_json'):
                            chart_key = f"chart_{uuid.uuid4().hex[:8]}"
                            st.plotly_chart(chart, use_container_width=True, key = chart_key)
                        elif hasattr(chart, 'figure'):  # Matplotlib chart
                            st.pyplot(chart.figure, use_container_width=True)
                        else:
                            # Try to display as plotly first, then matplotlib
                            try:
                                chart_key = f"chart_{uuid.uuid4().hex[:8]}"
                                st.plotly_chart(chart, use_container_width=True, key = chart_key)
                            except:
                                st.pyplot(chart, use_container_width=True)

                    except Exception as e:
                        st.error(f"Error displaying chart {i + 1}: {str(e)}")

                    st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

            # Export dashboard option
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("📥 Export Dashboard Data", use_container_width=True):
                    from datetime import datetime
                    import json

                    dashboard_data = {
                        "session_id": st.session_state.get('thread_id', 'N/A'),
                        "user_id": st.session_state.get('user_id', 'N/A'),
                        "total_charts": len(st.session_state.get('all_plots', [])),
                        "total_messages": len(st.session_state.get('chat_history', [])),
                        "exported_at": datetime.now().isoformat(),
                        "conversation_summary": [
                            msg["content"][:100] + "..." if len(msg.get("content", "")) > 100 else msg.get("content",
                                                                                                           "")
                            for msg in st.session_state.get('chat_history', [])
                            if msg.get("role") == "user"
                        ]
                    }
                    st.download_button(
                        "⬇️ Download Dashboard Summary",
                        data=json.dumps(dashboard_data, indent=2),
                        file_name=f"dashboard_{display_id}.json",
                        mime="application/json",
                        use_container_width=True
                    )
        else:
            st.markdown("""
            <div style="text-align: center; padding: 3rem; color: #4a5568; background: white; border-radius: 10px; margin: 1rem 0;">
                <h3>📊 No Charts Generated Yet</h3>
                <p>Start asking questions to generate beautiful visualizations!</p>
            </div>
            """, unsafe_allow_html=True)


def render_chat_interface():
    """Render the enhanced main chat interface"""
    # Enhanced Chat Header with perfect alignment
    st.markdown("""
    <div class="chat-header">
        <h1>🚚 Logistics Analytics Assistant</h1>
        <p>AI-powered insights for optimized logistics operations</p>
    </div>
    """, unsafe_allow_html=True)

    # Chat Container with perfect alignment to header
    # st.markdown('<div class="chat-container">', unsafe_allow_html=True) #Nitesh

    # Welcome message
    render_welcome_message()

    # Enhanced chat messages with better styling
    for i, message in enumerate(st.session_state.chat_history):
        role = message["role"]
        content = message["content"]
        img_list = message.get('img_list', [])
        timestamp = message.get('timestamp', '')

        if role == "user":
            # Enhanced user message
            st.markdown(f"""
            <div class="message-container user-message-container">
                <div class="user-message">
                    {content}
                    {f'<br><small style="opacity: 0.8;">📅 {timestamp}</small>' if timestamp else ''}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Display charts for user messages
            if img_list:
                st.markdown("**📊 Generated Charts:**")
                for idx_img, chart in enumerate(img_list):
                    display_chart(chart, idx_img, i, "user")

        else:  # bot message
            if content:
                # Parse agent name from content
                parts = content.split(':', 1)
                if len(parts) > 1:
                    agent_name = parts[0].replace('*', '').strip()
                    agent_content = parts[1].strip()
                else:
                    agent_name = "AI Assistant"
                    agent_content = content

                # Enhanced bot message
                st.markdown(f"""
                <div class="message-container bot-message-container">
                    <div class="bot-message">
                        <strong style="color: #667eea;">{agent_name}</strong><br>
                        {agent_content}
                        {f'<br><small style="opacity: 0.7;">📅 {timestamp}</small>' if timestamp else ''}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Display charts for bot messages
            if img_list:
                st.markdown("**📊 Analysis Charts:**")
                for idx_img, chart in enumerate(img_list):
                    display_chart(chart, idx_img, i, "bot")

    # Enhanced processing indicator
    if st.session_state.is_processing:
        render_typing_indicator()
        st.markdown("🔄 **Processing your request...** This may take a moment.")

    st.markdown('</div>', unsafe_allow_html=True)


def render_follow_up_questions():
    """Render enhanced follow-up questions"""
    if st.session_state.follow_up:
        st.markdown("""
        <div class="followup-container">
            <h4>💡 Suggested Next Questions</h4>
        </div>
        """, unsafe_allow_html=True)

        # Create responsive columns for follow-up questions
        cols = st.columns(min(len(st.session_state.follow_up), 3))
        for i, question in enumerate(st.session_state.follow_up):
            with cols[i % len(cols)]:
                if st.button(
                        question,
                        key=f"followup_{i}_{hash(question)}",
                        use_container_width=True,
                        help="Click to ask this question"
                ):
                    st.session_state.chat_input_text = question
                    st.rerun()


def render_feedback_section():
    """Render enhanced feedback section"""
    if st.session_state.chat_history:
        st.markdown("### 💬 How was my response?")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            feedback = st.feedback(
                "stars",
                key=f"feedback_{len(st.session_state.chat_history)}"
            )
            if feedback:
                st.session_state.feedback = feedback
                st.success("Thank you for your feedback! 🙏")


def render_input_area(data: Dict, azure_client, llm):
    """Render the enhanced fixed input area with perfect alignment"""
    # Enhanced input form with perfect alignment
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])

        with col1:
            user_input = st.text_area(
                "💬 Ask me anything about your logistics data...",
                value=st.session_state.chat_input_text,
                height=80,
                placeholder="Example: What are our top shipping routes by volume?",
                label_visibility="collapsed"
            )

        with col2:
            send_button = st.form_submit_button(
                "🚀 Send",
                use_container_width=True,
                type="primary"
            )

            clear_button = st.form_submit_button(
                "🗑️ Clear",
                use_container_width=True
            )

        # Enhanced quick action buttons
        st.markdown("**⚡ Quick Actions:**")
        quick_cols = st.columns(4)
        quick_actions = [
            "📊 Show dashboard",
            "💰 Cost analysis",
            "📈 Performance trends",
            "🎯 Optimization tips"
        ]

        for i, action in enumerate(quick_actions):
            with quick_cols[i]:
                if st.form_submit_button(action, use_container_width=True):
                    user_input = action.split(" ", 1)[1]  # Remove emoji

        # Process input
        if send_button and user_input.strip():
            st.session_state.is_processing = True
            st.session_state.show_welcome = False
            st.rerun()

        if clear_button:
            st.session_state.chat_history = []
            st.session_state.follow_up = []
            st.session_state.image = []
            st.session_state.all_plots = []  # Clear all plots
            st.session_state.show_welcome = True
            st.rerun()

    # Process the input after form submission
    if st.session_state.is_processing and user_input.strip():
        process_user_input(user_input.strip(), data, azure_client, llm)


def process_user_input(question: str, data: Dict, azure_client, llm):
    """Process user input and generate response with enhanced chart handling"""
    try:
        # Reset current image list
        st.session_state.image = []

        # Get AI response
        with st.spinner("🤖 Analyzing your request..."):
            result = get_result(question, data, azure_client, llm)

        # Add user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": question,
            "img_list": st.session_state.image.copy(),  # Store current plots with user message
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })

        # Process response
        bot_responses = prepare_final_text(result)

        # Add bot responses
        for bot_response in bot_responses:
            st.session_state.chat_history.append({
                "role": "bot",
                "content": bot_response,
                "img_list": [],
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })

        # Update follow-up questions
        st.session_state.follow_up = result.get('follow_up', [])

        # Reset processing state and clear input
        st.session_state.is_processing = False
        st.session_state.chat_input_text = ""

        st.rerun()

    except Exception as e:
        st.error(f"❌ Error processing your request: {str(e)}")
        st.session_state.is_processing = False
        st.rerun()


def handle_conversation_selection():
    """Handle conversation selection from sidebar"""
    if st.session_state.selected_thread:
        if st.session_state.selected_thread == "New_Conversation":
            st.session_state.thread_id = uuid.uuid4().hex
            st.session_state.chat_history = []
            st.session_state.follow_up = []
            st.session_state.image = []
            st.session_state.all_plots = []  # Reset plots for new conversation
            st.session_state.show_welcome = True
        else:
            st.session_state.thread_id = st.session_state.selected_thread
            st.session_state.chat_history = st.session_state.user_chat_history.get(
                st.session_state.selected_thread, []
            )
            st.session_state.show_welcome = False

        st.session_state.selected_thread = None
        st.rerun()


def main():
    """Enhanced main application function"""
    # Initialize session state
    initialize_session_state()

    # Load data and services
    data = load_data()
    azure_client, llm = initialize_services()

    if not data or not azure_client or not llm:
        st.error("❌ Failed to initialize application. Please check your configuration.")
        return

    # Handle conversation selection
    handle_conversation_selection()

    # Render dashboard modal if needed
    if st.session_state.show_dashboard:
        render_dashboard_modal()
        return  # Don't render main interface when dashboard is open

    # Create perfect layout with exact alignment
    col1, col2 = st.columns([1, 8])

    with col1:
        render_sidebar()

    with col2:
        # Main chat area with perfect alignment
        render_chat_interface()

        # Follow-up questions
        render_follow_up_questions()

        # Feedback section
        render_feedback_section()

    # Enhanced fixed input area with perfect alignment
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    st.markdown('<div class="input-container-content">', unsafe_allow_html=True)
    render_input_area(data, azure_client, llm)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Enhanced footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #4a5568; padding: 2rem; margin-bottom: 140px; font-weight: 500;'>"
        "🚚 Logistics Analytics Assistant | Powered by Advanced AI | "
        f"Session: {st.session_state.thread_id[:8]} | Charts Generated: {len(st.session_state.all_plots)}"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
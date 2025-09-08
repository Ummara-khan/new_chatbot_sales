import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import json
import os
import re
from datetime import datetime, timedelta
import hashlib
import openai
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
from collections import Counter
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv
import streamlit as st
from groq import Groq  # assuming you have the Groq SDK installed

# Load environment variables
load_dotenv()

# Try to import Groq (optional)
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# Configuration
UPLOAD_DIR = "uploads"
MAX_CONTEXT_LEN = 2000
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

# Create upload directory
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Page configuration
st.set_page_config(
    page_title="Analytics & AI Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #2c3e50;
        margin: 0;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #7f8c8d;
        margin: 0;
    }
    .metric-change {
        font-size: 0.8rem;
        font-weight: bold;
        margin-top: 0.5rem;
    }
    .metric-change.positive {
        color: #27ae60;
    }
    .metric-change.negative {
        color: #e74c3c;
    }
    .file-type-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        cursor: pointer;
        transition: transform 0.3s;
        margin: 1rem;
    }
    .file-type-card:hover {
        transform: translateY(-5px);
    }
    .chat-container {
        max-height: 400px;
        overflow-y: auto;
        padding: 1rem;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .user-message {
        display: flex;
        justify-content: flex-end;
        margin: 0.5rem 0;
    }
    .user-bubble {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.8rem 1.2rem;
        border-radius: 18px 18px 5px 18px;
        max-width: 70%;
        word-wrap: break-word;
    }
    .bot-message {
        display: flex;
        justify-content: flex-start;
        margin: 0.5rem 0;
    }
    .bot-bubble {
        background: #f8f9fa;
        color: #333;
        padding: 0.8rem 1.2rem;
        border-radius: 18px 18px 18px 5px;
        max-width: 70%;
        word-wrap: break-word;
        border: 1px solid #e9ecef;
    }
    .dashboard-section {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    .chart-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .file-history-item {
        padding: 0.5rem;
        margin: 0.2rem 0;
        border-radius: 5px;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .file-history-item:hover {
        background-color: #f0f0f0;
    }
    .file-history-item.active {
        background-color: #667eea;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'
if 'file_type' not in st.session_state:
    st.session_state.file_type = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'uploaded_files_history' not in st.session_state:
    st.session_state.uploaded_files_history = []

# Simple user database (in production, use proper database)
USERS = {
    "admin": "5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8",  # "password"
    "user": "ef92b778bafe771e89245b89ecbc08a44a4e166c06659911881f383d4473e94f",   # "secret"
    "demo": "2bb80d537b1da3e38bd30361aa855686bde0eacd7162fef6a25fe97bf527a25b"    # "demo123"
}

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def authenticate(username, password):
    return username in USERS and USERS[username] == hash_password(password)

# Initialize Groq client


# Get Groq API key from .env
GROQ_API_KEY = api_key=st.secrets["GROQ_API_KEY"]

groq_client = None
try:
    if GROQ_API_KEY:
        groq_client = Groq(api_key=GROQ_API_KEY)
    else:
        raise ValueError("Groq API key not found in environment")
except Exception as e:
    st.error(f"⚠️ Groq API key not configured. Some AI features may not work.\n{e}")
    groq_client = None


# ----------------- HELPERS -----------------
# ----------------- CONFIG -----------------
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
openai.api_key = st.secrets["OPENAI_API_KEY"]

# ----------------- HELPER FUNCTIONS -----------------
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'<.*?>', '', text)
    return text.strip()


def xml_to_rows(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    rows = []
    def parse_element(elem, parent_path=""):
        path = f"{parent_path}/{elem.tag}" if parent_path else elem.tag
        if elem.text and elem.text.strip():
            rows.append({'path': path, 'text': elem.text.strip()})
        for child in elem:
            parse_element(child, path)
    parse_element(root)
    return pd.DataFrame(rows)

def html_to_rows(html_file):
    soup = BeautifulSoup(html_file.read(), 'html.parser')
    texts = [clean_text(t) for t in soup.stripped_strings if t.strip()]
    return pd.DataFrame({'path': ['html'] * len(texts), 'text': texts})

def txt_to_rows(txt_file):
    lines = []
    for line in txt_file.readlines():
        if isinstance(line, bytes):
            line = line.decode('utf-8', errors='ignore')
        line = clean_text(line)
        if line.strip():
            lines.append(line)
    return pd.DataFrame({'path': ['txt'] * len(lines), 'text': lines})

def csv_to_rows(csv_file):
    df = pd.read_csv(csv_file)
    return pd.DataFrame({'path': df.columns.repeat(len(df)), 'text': df.astype(str).values.flatten()})

def json_to_rows(json_file):
    df = pd.read_json(json_file)
    return pd.DataFrame({'path': df.columns.repeat(len(df)), 'text': df.astype(str).values.flatten()})

def excel_to_rows(excel_file):
    df = pd.read_excel(excel_file)
    return pd.DataFrame({'path': df.columns.repeat(len(df)), 'text': df.astype(str).values.flatten()})

def load_file(uploaded_file):
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    if ext == '.xml':
        return xml_to_rows(uploaded_file)
    elif ext in ['.html', '.htm']:
        return html_to_rows(uploaded_file)
    elif ext == '.txt':
        return txt_to_rows(uploaded_file)
    elif ext == '.csv':
        return csv_to_rows(uploaded_file)
    elif ext == '.json':
        return json_to_rows(uploaded_file)
    elif ext in ['.xls', '.xlsx']:
        return excel_to_rows(uploaded_file)
    else:
        st.error("Unsupported file type")
        return pd.DataFrame()

def extract_numbers(text):
    return [float(x.replace(',', '')) for x in re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', text)]

def generate_human_readable_summary(numbers, query, context_texts=None):
    """
    Generate a specific, human-readable answer using OpenAI.
    If query relates to hotel data, try to extract fields like rooms, area, revenue.
    """
    if not numbers and not context_texts:
        return "No relevant data found for your query.", None

    # Provide LLM with more context
    context_sample = " ".join(context_texts[:5]) if context_texts else "No extra context"
    numbers_str = ', '.join([f"{n:.2f}" for n in numbers[:50]])  # limit numbers

    prompt = f"""
    You are analyzing structured hotel dataset values.
    User query: '{query}'
    Extracted numbers: {numbers_str}
    Context: {context_sample}

    Instructions:
    - If the query mentions hotel details (like "hotel 5", "rooms", "area", "revenue"), 
      give a specific answer in plain English.
    - Example output style:
        "Hotel 5 has 8 rooms and generated 599,523.75 in revenue."
    - Do not just list numbers; map them to meaning (rooms, revenue, area) if context allows.
    - If uncertain, give your best interpretation based on context.
    - Keep it short and precise.
    """

    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=200
        )
        summary = response.choices[0].message.content.strip()
    except Exception as e:
        summary = f"⚠️ Error generating summary: {e}"

    # Auto-generate chart if query asks for it
    if any(word in query.lower() for word in ["plot", "graph", "chart", "visualize"]):
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(numbers, marker="o", linestyle="-")
        ax.set_title("Hotel Data Visualization", fontsize=10)
        ax.set_xlabel("Index", fontsize=8)
        ax.set_ylabel("Value", fontsize=8)
        plt.tight_layout()
        return summary, fig
    else:
        return summary, None
    



def rewrite_query(query):
    """Use Groq to rewrite the query into a clean version"""
    if groq_client is None:
        return query
    try:
        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": "You are a query rewriter. Fix typos and slang, output only cleaned query."},
                {"role": "user", "content": query}
            ],
            max_completion_tokens=64
        )
        return completion.choices[0].message.content.strip()
    except Exception:
        return query

def query_bot_response(df=None, pdf_text="", query=""):
    """Returns a human-readable bot-style response for data or PDF queries"""
    if groq_client is None:
        return "🤖 AI service is currently unavailable. Please configure your Groq API key."
        
    context_summary = ""
    
    try:
        if df is not None and not df.empty:
            numeric_cols = df.select_dtypes(include="number").columns
            categorical_cols = df.select_dtypes(exclude="number").columns
            
            for col in numeric_cols[:5]:  # Limit to prevent context overflow
                try:
                    top_item = df.sort_values(col, ascending=False).iloc[0]
                    text_col = categorical_cols[0] if len(categorical_cols) > 0 else None
                    item_name = top_item[text_col] if text_col and text_col in top_item else f"Row {top_item.name}"
                    context_summary += f"Column '{col}': top item is '{item_name}' with value {top_item[col]}\n"
                except:
                    continue
            
            for col in categorical_cols[:5]:  # Limit to prevent context overflow
                try:
                    top_cat = df[col].value_counts().head(3)
                    context_summary += f"Column '{col}': top categories are {dict(top_cat)}\n"
                except:
                    continue
        
        elif pdf_text:
            paragraphs = [p.strip() for p in pdf_text.split("\n") if p.strip()]
            counter = Counter(" ".join(paragraphs).split())
            top_words = [w for w,c in counter.most_common(20)]
            relevant_paragraphs = [p for p in paragraphs if any(w in p for w in top_words)]
            context_summary = "\n".join(relevant_paragraphs)[:MAX_CONTEXT_LEN]
        
        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",  # Use available model instead of openai/gpt-oss-20b
            messages=[
                {"role":"system", "content":"You are a smart data analyst bot. Provide human-readable, concise answers. Reason about numeric/categorical trends. Summarize PDFs if needed."},
                {"role":"user", "content": f"Context:\n{context_summary}\n\nQuestion: {query}"}
            ],
            max_completion_tokens=512
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"❌ Error processing your query: {str(e)}"

# Login page
def show_login():
    st.markdown('<div class="main-header"><h1>🔐 Login to Analytics Platform</h1></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### Please enter your credentials")
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        
        if st.button("Login", use_container_width=True):
            if authenticate(username, password):
                st.session_state.authenticated = True
                st.session_state.username = username
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid credentials!")
        
        st.markdown("---")
        st.info("Demo credentials:\n- Username: demo, Password: demo123\n- Username: admin, Password: password")

# Home page after login
def show_home():
    st.markdown(f'<div class="main-header"><h1>Welcome, {st.session_state.username}! 📊</h1><p>Choose your data analysis approach</p></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Structured Files", use_container_width=True, help="CSV, Excel, JSON files"):
            st.session_state.file_type = 'structured'
            st.session_state.current_page = 'file_upload'
            st.rerun()
        st.markdown("""
        <div style="text-align: center; margin-top: 1rem;">
            <p>📈 CSV, Excel, JSON</p>
            <p>Perfect for tabular data analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if st.button("📄 Unstructured Files", use_container_width=True, help="PDF, TXT, HTML, XML files"):
            st.session_state.file_type = 'unstructured'
            st.session_state.current_page = 'file_upload'
            st.rerun()
        st.markdown("""
        <div style="text-align: center; margin-top: 1rem;">
            <p>📄 PDF, TXT, HTML, XML</p>
            <p>AI-powered document analysis</p>
        </div>
        """, unsafe_allow_html=True)

def show_file_upload():
    file_type_name = "Structured" if st.session_state.file_type == 'structured' else "Unstructured"
    st.markdown(f'<div class="main-header"><h1>{file_type_name} File Upload 📁</h1><p>Upload your files to get started</p></div>', unsafe_allow_html=True)
    
    # File upload section
    if st.session_state.file_type == 'structured':
        uploaded_files = st.file_uploader(
            "Upload structured files",
            type=["csv", "xlsx", "json"],
            accept_multiple_files=True,
            help="Upload CSV, Excel, or JSON files"
        )
    else:
        uploaded_files = st.file_uploader(
            "Upload unstructured files",
            type=["pdf", "txt", "html", "xml"],
            accept_multiple_files=True,
            help="Upload PDF, TXT, HTML, or XML files"
        )
    
    if uploaded_files:
        # Store uploaded files in session state
        if 'uploaded_files_data' not in st.session_state:
            st.session_state.uploaded_files_data = {}
        
        # Process files
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.uploaded_files_data:
                file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                df, text_data = None, ""
                
                try:
                    if st.session_state.file_type == 'structured':
                        if uploaded_file.name.lower().endswith(".csv"):
                            df = pd.read_csv(file_path)
                        elif uploaded_file.name.lower().endswith(".xlsx"):
                            df = pd.read_excel(file_path)
                        elif uploaded_file.name.lower().endswith(".json"):
                            with open(file_path, 'r') as f:
                                data = json.load(f)
                            if isinstance(data, list):
                                df = pd.DataFrame(data)
                            elif isinstance(data, dict):
                                for v in data.values():
                                    if isinstance(v, list):
                                        df = pd.DataFrame(v)
                                        break
                    else:
                        if uploaded_file.name.lower().endswith(".pdf"):
                            reader = PdfReader(file_path)
                            text_data = "\n".join([p.extract_text() or "" for p in reader.pages])
                        elif uploaded_file.name.lower().endswith(".xml"):
                            df = xml_to_rows(file_path)
                            text_data = " ".join(df['text'].tolist()) if not df.empty else ""
                        elif uploaded_file.name.lower().endswith((".html", ".htm")):
                            with open(file_path, 'rb') as f:
                                df = html_to_rows(f)
                            text_data = " ".join(df['text'].tolist()) if not df.empty else ""
                        elif uploaded_file.name.lower().endswith(".txt"):
                            with open(file_path, 'rb') as f:
                                df = txt_to_rows(f)
                            text_data = " ".join(df['text'].tolist()) if not df.empty else ""
                    
                    st.session_state.uploaded_files_data[uploaded_file.name] = {"df": df, "text": text_data}
                    st.success(f"✅ Processed: {uploaded_file.name}")
                    
                    # Update file history
                    history_entry = (uploaded_file.name, st.session_state.file_type)
                    if 'uploaded_files_history' not in st.session_state:
                        st.session_state.uploaded_files_history = [history_entry]
                    else:
                        # Check if the file is already in history
                        existing_files = [f[0] for f in st.session_state.uploaded_files_history]
                        if uploaded_file.name not in existing_files:
                            st.session_state.uploaded_files_history.append(history_entry)
                    
                except Exception as e:
                    st.error(f"❌ Error processing {uploaded_file.name}: {e}")
        
        st.markdown("---")
        st.markdown("### Choose your analysis approach:")
        
        if st.session_state.file_type == 'structured':
            # Two columns: Dashboard + Chatbot
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📊 Dashboard Analytics", use_container_width=True):
                    st.session_state.current_page = 'dashboard'
                    st.rerun()
                st.markdown("""
                <div style="text-align: center; margin-top: 1rem;">
                    <p>📈 Interactive Charts</p>
                    <p>Statistical Analysis & Insights</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                if st.button("💬 AI Chatbot", use_container_width=True):
                    st.session_state.current_page = 'chatbot'
                    st.rerun()
                st.markdown("""
                <div style="text-align: center; margin-top: 1rem;">
                    <p>🤖 Ask Questions</p>
                    <p>Get AI-powered insights</p>
                </div>
                """, unsafe_allow_html=True)
        
        else:
            # Only chatbot option (centered)
            col = st.columns([1, 2, 1])[1]  # center column
            with col:
                if st.button("💬 AI Chatbot", use_container_width=True):
                    st.session_state.current_page = 'chatbot'
                    st.rerun()
                st.markdown("""
                <div style="text-align: center; margin-top: 1rem;">
                    <p>🤖 Ask Questions</p>
                    <p>Get AI-powered insights from unstructured files</p>
                </div>
                """, unsafe_allow_html=True)
    
    if st.button("← Back to Home"):
        st.session_state.current_page = 'home'
        st.rerun()


def show_dashboard():
    file_type_name = "Structured" if st.session_state.file_type == 'structured' else "Unstructured"
    st.markdown(f'<div class="main-header"><h1>📊 {file_type_name} Analytics</h1></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("💬 Switch to Chatbot", use_container_width=True):
            st.session_state.current_page = 'chatbot'
            st.rerun()
    with col2:
        if st.button("← Back to Upload", use_container_width=True):
            st.session_state.current_page = 'file_upload'
            st.rerun()

    if 'uploaded_files_data' in st.session_state and st.session_state.uploaded_files_data:
        loaded_data = st.session_state.uploaded_files_data

        # File selector
        selected_file = st.selectbox("📂 Select file to analyze", list(loaded_data.keys()))
        df = loaded_data[selected_file]["df"]
        text_data = loaded_data[selected_file]["text"]

        # Only show dashboard for structured data
        if st.session_state.file_type == 'structured' and df is not None:
            show_structured_analytics(df, selected_file)
        else:
            st.info("📄 Unstructured file detected — Switching to Chatbot mode...")
            st.session_state.current_page = 'chatbot'
            st.rerun()
    else:
        st.warning("No files uploaded. Please go back and upload files first.")
        if st.button("← Back to Upload"):
            st.session_state.current_page = 'file_upload'
            st.rerun()


# Cache model + index
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)

@st.cache_resource
def build_faiss_index(texts):
    model = load_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=False, batch_size=32)
    embeddings = np.array(embeddings).astype('float32')
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings


def show_chatbot():
    file_type_name = "Structured" if st.session_state.file_type == 'structured' else "Unstructured"
    st.markdown(f'<div class="main-header"><h1>💬 {file_type_name} AI Chatbot</h1></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("📊 Switch to Dashboard", use_container_width=True):
            st.session_state.current_page = 'dashboard'
            st.rerun()
    with col2:
        if st.button("← Back to Upload", use_container_width=True):
            st.session_state.current_page = 'file_upload'
            st.rerun()
    
    if 'uploaded_files_data' in st.session_state and st.session_state.uploaded_files_data:
        loaded_data = st.session_state.uploaded_files_data
        
        # File selector for chat
        selected_file = st.selectbox("📂 Select file to chat about", list(loaded_data.keys()))
        df = loaded_data[selected_file]["df"]
        text_data = loaded_data[selected_file]["text"]
        
        st.success(f"✅ Chatting about: {selected_file}")
        
        st.markdown("### 💬 Chat with your data")
        
        # Chat container with improved styling
        chat_container = st.container()
        with chat_container:
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            
            # Display chat history with proper bubbles
            for i, (role, message) in enumerate(st.session_state.chat_history):
                if role == "user":
                    st.markdown(f'''
                    <div class="user-message">
                        <div class="user-bubble">
                            <strong>You:</strong> {message}
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
                else:
                    st.markdown(f'''
                    <div class="bot-message">
                        <div class="bot-bubble">
                            <strong>🤖 Assistant:</strong> {message}
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Chat input
        user_query = st.text_input("Ask a question about your data:", placeholder="What insights can you find in this data?")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("Send", use_container_width=True):
                if user_query:
                    # Add user message to history
                    st.session_state.chat_history.append(("user", user_query))
                    
                    # Get bot response
                    with st.spinner("Thinking..."):
                        if st.session_state.file_type == 'unstructured' and text_data and df is not None and not df.empty:
                            # For unstructured files, use semantic search
                            try:
                                model = load_embedding_model()
                                index, embeddings = build_faiss_index(df['text'].tolist())
                                
                                query_vec = model.encode([user_query]).astype('float32')
                                D, I = index.search(query_vec, k=5)
                                
                                top_texts = [df.iloc[idx]['text'] for idx in I[0]]
                                context = "\n".join(top_texts)
                                
                                response = query_bot_response(pdf_text=context, query=user_query)
                            except Exception as e:
                                response = f"Error in semantic search: {str(e)}"
                        else:
                            # For structured files or simple text
                            response = query_bot_response(df=df, pdf_text=text_data, query=user_query)
                    
                    # Add bot response to history
                    st.session_state.chat_history.append(("bot", response))
                    st.rerun()
                else:
                    st.warning("Please enter a question!")
        
        with col2:
            if st.button("Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
    else:
        st.warning("No files uploaded. Please go back and upload files first.")
        if st.button("← Back to Upload"):
            st.session_state.current_page = 'file_upload'
            st.rerun()

def show_structured_analytics(df, filename):
    """Display analytics dashboard for structured data with 12 visualizations"""
    st.markdown(f"### 📊 Analytics Dashboard - {filename}")
    
    if df is None or df.empty:
        st.warning("No data available for analysis")
        return
    
    # Get numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # Generate AI insights using Groq
    if groq_client:
        try:
            summary = df.describe(include="all").to_string()
            prompt = f"Here is a dataset summary. Give 2-3 business insights in simple language:\n{summary}"
            response = groq_client.chat.completions.create(
                model="openai/gpt-oss-20b",  # Use available model
                messages=[{"role": "user", "content": prompt}]
            )
            ai_insights = response.choices[0].message.content
            st.info(f"🤖 AI Insights: {ai_insights}")
        except Exception as e:
            st.warning("AI insights temporarily unavailable")
    
    # Row 1: 3 KPI Cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if numeric_cols:
            max_val = df[numeric_cols[0]].max()
            col_name = numeric_cols[0]
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-label">Highest {col_name}</p>
                <p class="metric-value">{max_val:,.0f}</p>
                <p class="metric-change positive">Peak Value</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-label">Total Records</p>
                <p class="metric-value">{len(df):,}</p>
                <p class="metric-change positive">Complete</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if len(numeric_cols) >= 2:
            avg_val = df[numeric_cols[1]].mean()
            col_name = numeric_cols[1]
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-label">Average {col_name}</p>
                <p class="metric-value">{avg_val:,.1f}</p>
                <p class="metric-change positive">Mean Value</p>
            </div>
            """, unsafe_allow_html=True)
        elif numeric_cols:
            avg_val = df[numeric_cols[0]].mean()
            col_name = numeric_cols[0]
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-label">Average {col_name}</p>
                <p class="metric-value">{avg_val:,.1f}</p>
                <p class="metric-change positive">Mean Value</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            unique_count = df[categorical_cols[0]].nunique() if categorical_cols else len(df.columns)
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-label">Data Fields</p>
                <p class="metric-value">{unique_count}</p>
                <p class="metric-change positive">Columns</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        if numeric_cols:
            total_val = df[numeric_cols[0]].sum()
            col_name = numeric_cols[0]
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-label">Total {col_name}</p>
                <p class="metric-value">{total_val:,.0f}</p>
                <p class="metric-change positive">Sum</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-label">Data Quality</p>
                <p class="metric-value">{completeness:.1f}%</p>
                <p class="metric-change positive">Complete</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        if numeric_cols:
            st.markdown(f"#### 📊 {numeric_cols[0]} Distribution")
            fig = px.histogram(df, x=numeric_cols[0], nbins=20, color_discrete_sequence=['#667eea'])
            fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', height=300)
            st.plotly_chart(fig, use_container_width=True)
        elif categorical_cols:
            st.markdown(f"#### 📊 {categorical_cols[0]} Count")
            value_counts = df[categorical_cols[0]].value_counts().head(10)
            fig = px.bar(x=value_counts.index, y=value_counts.values, color_discrete_sequence=['#667eea'])
            fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', height=300)
            st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        if len(numeric_cols) >= 2:
            st.markdown(f"#### 📈 {numeric_cols[0]} vs {numeric_cols[1]}")
            fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], color_discrete_sequence=['#764ba2'])
            fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', height=300)
            st.plotly_chart(fig, use_container_width=True)
        elif numeric_cols:
            st.markdown(f"#### 📈 {numeric_cols[0]} Trend")
            fig = px.line(df.head(50), x=df.head(50).index, y=numeric_cols[0], color_discrete_sequence=['#764ba2'])
            fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', height=300)
            st.plotly_chart(fig, use_container_width=True)
        elif categorical_cols:
            st.markdown(f"#### 🥧 {categorical_cols[0]} Distribution")
            value_counts = df[categorical_cols[0]].value_counts().head(8)
            fig = px.pie(values=value_counts.values, names=value_counts.index, color_discrete_sequence=px.colors.sequential.Purples_r)
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        if len(numeric_cols) >= 3:
            st.markdown(f"#### 📊 {numeric_cols[2]} Analysis")
            fig = px.box(df, y=numeric_cols[2], color_discrete_sequence=['#f093fb'])
            fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', height=300)
            st.plotly_chart(fig, use_container_width=True)
        elif len(categorical_cols) >= 2:
            st.markdown(f"#### 📊 {categorical_cols[1]} Analysis")
            value_counts = df[categorical_cols[1]].value_counts().head(10)
            fig = px.bar(x=value_counts.values, y=value_counts.index, orientation='h', color_discrete_sequence=['#f093fb'])
            fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("#### 📊 Data Overview")
            missing_data = df.isnull().sum()
            fig = px.bar(x=missing_data.values, y=missing_data.index, orientation='h', 
                        title="Missing Values by Column", color_discrete_sequence=['#f093fb'])
            fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', height=300)
            st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        if numeric_cols:
            st.markdown(f"#### 📊 {numeric_cols[0]} Statistics")
            stats_data = df[numeric_cols[0]].describe()
            fig = px.bar(x=['Min', 'Q1', 'Median', 'Q3', 'Max'], 
                        y=[stats_data['min'], stats_data['25%'], stats_data['50%'], stats_data['75%'], stats_data['max']],
                        color_discrete_sequence=['#667eea'])
            fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("#### 📊 Column Types")
            col_types = {'Numeric': len(numeric_cols), 'Categorical': len(categorical_cols)}
            fig = px.pie(values=list(col_types.values()), names=list(col_types.keys()), 
                        color_discrete_sequence=['#667eea', '#764ba2'])
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        if len(numeric_cols) >= 2:
            st.markdown("#### 🔥 Correlation Heatmap")
            corr_matrix = df[numeric_cols[:5]].corr()
            fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='Purples')
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        elif categorical_cols:
            st.markdown(f"#### 📊 Top {categorical_cols[0]} Values")
            top_values = df[categorical_cols[0]].value_counts().head(6)
            fig = px.pie(values=top_values.values, names=top_values.index, 
                        color_discrete_sequence=px.colors.sequential.Purples_r)
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        if numeric_cols and len(df) > 10:
            st.markdown(f"#### 📈 {numeric_cols[0]} Moving Average")
            df_sorted = df.sort_values(by=numeric_cols[0]) if numeric_cols[0] in df.columns else df
            moving_avg = df_sorted[numeric_cols[0]].rolling(window=min(5, len(df)//2)).mean()
            fig = px.line(x=range(len(moving_avg)), y=moving_avg, 
                         title=f"Moving Average Trend", color_discrete_sequence=['#f093fb'])
            fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("#### 📊 Data Completeness")
            completeness_by_col = (1 - df.isnull().sum() / len(df)) * 100
            fig = px.bar(x=completeness_by_col.index, y=completeness_by_col.values,
                        title="Completeness by Column", color_discrete_sequence=['#f093fb'])
            fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', height=300)
            st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        if numeric_cols:
            st.markdown(f"#### 📊 {numeric_cols[0]} Outliers")
            fig = px.violin(df, y=numeric_cols[0], color_discrete_sequence=['#8e44ad'])
            fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("#### 📊 Data Types")
            type_counts = df.dtypes.value_counts()
            fig = px.bar(x=type_counts.index.astype(str), y=type_counts.values, color_discrete_sequence=['#8e44ad'])
            fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', height=300)
            st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        if len(numeric_cols) >= 2:
            st.markdown(f"#### 📈 {numeric_cols[1]} Distribution")
            fig = px.histogram(df, x=numeric_cols[1], nbins=15, color_discrete_sequence=['#9b59b6'])
            fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', height=300)
            st.plotly_chart(fig, use_container_width=True)
        elif categorical_cols and len(categorical_cols) >= 2:
            st.markdown(f"#### 📊 {categorical_cols[1]} Breakdown")
            value_counts = df[categorical_cols[1]].value_counts().head(8)
            fig = px.pie(values=value_counts.values, names=value_counts.index, color_discrete_sequence=px.colors.sequential.Purples)
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        if numeric_cols:
            st.markdown(f"#### 📊 Data Range Analysis")
            ranges = []
            for col in numeric_cols[:3]:
                ranges.append({'Column': col, 'Range': df[col].max() - df[col].min()})
            if ranges:
                range_df = pd.DataFrame(ranges)
                fig = px.bar(range_df, x='Column', y='Range', color_discrete_sequence=['#a569bd'])
                fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', height=300)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("#### 📊 Missing Data Pattern")
            missing_pattern = df.isnull().sum()
            fig = px.pie(values=missing_pattern.values, names=missing_pattern.index, color_discrete_sequence=px.colors.sequential.Purples)
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

import streamlit as st
import plotly.express as px

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import openai

openai.api_key = st.secrets["OPENAI_API_KEY"]

# ----------------- Parse with OpenAI -----------------
def parse_with_openai(text_data, filename):
    """Use OpenAI to convert unstructured text (xml, html, txt) into structured JSON."""
    prompt = f"""
    You are a data extraction assistant.
    Extract structured data from the following {filename} content.
    Return it as JSON with keys for numeric and categorical fields where possible.

    Content:
    {text_data[:5000]}  # limit to avoid overflow
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # or gpt-4.1-mini
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        structured_text = response["choices"][0]["message"]["content"]
        structured_data = json.loads(structured_text)  # Parse JSON
        return pd.DataFrame(structured_data)

    except Exception as e:
        st.error(f"❌ OpenAI parsing failed: {e}")
        return None


# ----------------- Dashboard -----------------


# ----------------- Usage -----------------
import streamlit as st

# Function to process the uploaded file
def process_file(uploaded_file):
    # Read file content (adjust for structured/unstructured)
    text_data = uploaded_file.read().decode("utf-8")
    return text_data

# Sidebar logic
with st.sidebar:
    if st.session_state.get('authenticated', False):
        st.markdown(f"### 👤 Welcome, {st.session_state.username}")

        if st.button("🏠 Home", use_container_width=True):
            st.session_state.current_page = 'home'
            st.rerun()

        # Display current session file
        if st.session_state.get('current_file_name'):
            st.markdown(f"### 📂 Current File: {st.session_state.current_file_name}")

        # Logout
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.clear()  # Clears everything from session_state
            st.session_state.authenticated = False
            st.session_state.current_page = 'home'
            st.rerun()

        st.markdown("---")
        st.markdown("### 📊 Current Session Info")
        st.info(f"File Type: {st.session_state.get('file_type', 'None')}")
        st.info(f"Page: {st.session_state.get('current_page', 'home')}")

        if 'groq_client' in st.session_state:
            st.success("🤖 AI Assistant: Ready")
        else:
            st.warning("🤖 AI Assistant: Unavailable")

# Main app logic
if not st.session_state.get('authenticated', False):
    show_login()
else:
    if st.session_state.get('current_page') == 'home':
        show_home()

    elif st.session_state.get('current_page') == 'file_upload':
        uploaded_file = show_file_upload()  # Your existing file upload widget
        if uploaded_file:
            # Process file
            text_data = process_file(uploaded_file)

            # Store only for this current file
            st.session_state.current_file_name = uploaded_file.name
            st.session_state.current_file_type = st.session_state.get('file_type')
            st.session_state.current_file_data = text_data

            # Clear previous chat history
            st.session_state.chat_history = []

            # Move to dashboard or chatbot
            st.session_state.current_page = 'dashboard'
            st.rerun()

    elif st.session_state.get('current_page') == 'dashboard':
        show_dashboard(file_data=st.session_state.get('current_file_data'))

    elif st.session_state.get('current_page') == 'chatbot':
        show_chatbot(file_data=st.session_state.get('current_file_data'))



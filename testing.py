import streamlit as st
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re
import json
from groq import Groq

# ------------------------------
# CONFIG
# ------------------------------
st.set_page_config(page_title="Readable XML/HTML Chatbot", layout="wide")
st.title("📄 Offline XML/HTML Chatbot")

# Initialize Groq client
import streamlit as st

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

client = Groq(api_key=GROQ_API_KEY)

# ------------------------------
# Load embedding model
# ------------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

# ------------------------------
# Flatten XML/HTML with paths
# ------------------------------
def flatten_with_paths(element, parent_path=""):
    flat = {}
    for child in element.find_all(recursive=False):
        path = f"{parent_path}/{child.name}" if parent_path else child.name
        if child.find_all(recursive=False):
            flat.update(flatten_with_paths(child, path))
        else:
            flat[path] = child.get_text(strip=True)
    return flat

def flatten_xml_html(file_content, file_type="xml"):
    if file_type == "xml":
        soup = BeautifulSoup(file_content, "xml")
    else:
        soup = BeautifulSoup(file_content, "html.parser")
    flattened = []
    for r in soup.find_all(recursive=False):
        flattened.append(flatten_with_paths(r))
    return flattened

# ------------------------------
# Numeric Extraction Logic
# ------------------------------
def extract_numeric(flattened_chunks, query, agg="sum"):
    query_words = [w.lower() for w in re.findall(r"\w+", query)]
    values = []

    for chunk in flattened_chunks:
        if isinstance(chunk, dict):
            for key, value in chunk.items():
                if any(q in key.lower() or q in str(value).lower() for q in query_words):
                    try:
                        values.append(float(value))
                    except:
                        continue

    if not values:
        return None

    if agg == "sum":
        return f"Total {query}: {int(sum(values))}"
    elif agg == "max":
        return f"Highest {query}: {int(max(values))}"
    else:
        return f"{agg} of {query}: {values}"

# ------------------------------
# Groq Answering
# ------------------------------
def ask_groq(user_query, retrieved_chunks, numeric_answer=None):
    if numeric_answer:
        system_prompt = "You are a helpful assistant. Rewrite the extracted numeric answer in a clear, natural way."
        user_message = f"Query: {user_query}\nAnswer: {numeric_answer}"
    else:
        system_prompt = (
            "You are a helpful assistant. Answer the query using only the retrieved context below. "
            "If no clear answer, say 'No relevant data found.'"
            "if ask about adr use revenue and give total value of it"
        )

        # Compact retrieved chunks
        compact_context = []
        for ch in retrieved_chunks:
            compact_context.append(", ".join([f"{k}: {v}" for k, v in list(ch.items())[:6]]))  # limit 6 fields
        context_str = "\n".join(compact_context)

        user_message = f"Query: {user_query}\nContext:\n{context_str}"

    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.7,
        max_completion_tokens=512,  # ⬅ reduce tokens
        top_p=1,
        stream=False
    )

    return completion.choices[0].message.content



# ------------------------------
# Persistent Index (All Records)
# ------------------------------
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
    st.session_state.flattened_records = []

uploaded_file = st.file_uploader("Upload XML/HTML file", type=["xml", "html"])

if uploaded_file:
    file_content = uploaded_file.read()
    file_type = uploaded_file.type.split("/")[-1]

    flattened_data = flatten_xml_html(file_content, file_type)
    st.session_state.flattened_records.extend(flattened_data)

    # Update FAISS index for ALL records
    chunks_for_embedding = [str(record) for record in st.session_state.flattened_records]
    embeddings = embedding_model.encode(chunks_for_embedding, convert_to_numpy=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    st.session_state.faiss_index = index

    st.success(f"Total {len(st.session_state.flattened_records)} records indexed!")

# ------------------------------
# Query
# ------------------------------
user_query = st.text_input("Ask a question about your data:")

if user_query and st.session_state.faiss_index is not None:
    query_vec = embedding_model.encode([user_query])
    D, I = st.session_state.faiss_index.search(query_vec, k=5)
    relevant_chunks = [st.session_state.flattened_records[i] for i in I[0]]

    # Try numeric extraction
    numeric_answer = extract_numeric(relevant_chunks, user_query, agg="sum")
    if not numeric_answer:
        numeric_answer = extract_numeric(relevant_chunks, user_query, agg="max")

    # Ask Groq
    final_answer = ask_groq(user_query, relevant_chunks, numeric_answer)

    st.markdown("### Answer:")
    st.markdown(final_answer)

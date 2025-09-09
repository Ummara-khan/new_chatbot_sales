import os
import json
import streamlit as st
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
from openai import OpenAI

# -----------------------------
# Init OpenAI
# -----------------------------
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", None)

if not OPENAI_API_KEY:
    st.error("⚠️ Please set your OpenAI API key in .streamlit/secrets.toml")
else:
    client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------
# File Parsing
# -----------------------------
def parse_xml(content):
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return []

    def xml_to_dict(elem):
        d = {}
        for child in elem:
            if len(child) > 0:
                d[child.tag] = xml_to_dict(child)
            else:
                d[child.tag] = child.text or ""
        return d
    return [xml_to_dict(root)]

def parse_html(content):
    soup = BeautifulSoup(content, "html.parser")
    return [{"text": soup.get_text(separator=" ", strip=True)}]

def parse_txt(content):
    return [{"text": content}]

def parse_file(uploaded_file):
    content = uploaded_file.read().decode("utf-8", errors="ignore")
    name = uploaded_file.name.lower()
    if name.endswith(".xml"):
        return parse_xml(content)
    elif name.endswith((".html", ".htm")):
        return parse_html(content)
    elif name.endswith(".txt"):
        return parse_txt(content)
    else:
        return []

# -----------------------------
# GPT Query (batching support)
# -----------------------------
def ask_gpt(query, records, batch_size=50):
    """
    Use GPT to answer queries about any dataset (any structure).
    Splits records into batches, gets partial answers, then aggregates.
    """
    batch_answers = []
    total_records = len(records)

    # Step 1: Process batches
    for i in range(0, total_records, batch_size):
        batch = records[i:i+batch_size]
        context = json.dumps(batch, indent=2)

        prompt = f"""
You are a precise data assistant. 
You are given structured records from a file (XML/HTML/TXT converted to JSON). 
Always answer based ONLY on the provided context. 

⚠️ Rules:
- If the user asks "how many", try to count matching records.
- If the answer is numeric (like counts, totals), return the number clearly.
- If information is not in this batch, just say "Not found in this batch".

User Query:
{query}

Context (records {i+1}–{i+len(batch)} of {total_records}):
{context}

Answer (concise, factual, no guessing):
"""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        ans = response.choices[0].message.content.strip()
        batch_answers.append(ans)

    # Step 2: Aggregate all partial answers
    combined_context = "\n".join(batch_answers)

    final_prompt = f"""
The user query was:
{query}

Here are partial answers from multiple batches:
{combined_context}

Now combine them into ONE final clear answer. 
- If numbers were reported, add them up to give a total.
- If text answers were reported, summarize concisely.
- If nothing relevant was found, just say "Not found".
"""

    final_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": final_prompt}]
    )
    return final_response.choices[0].message.content

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="📂 Universal File Chatbot", layout="wide")
st.title("📂 Universal File Chatbot (XML / HTML / TXT)")

uploaded_file = st.file_uploader("📂 Upload your file", type=["xml", "html", "htm", "txt"])

if uploaded_file:
    with st.spinner("🔄 Parsing file..."):
        records = parse_file(uploaded_file)

    if not records:
        st.error("❌ Unsupported or empty file.")
    else:
        st.success(f"✅ File processed with {len(records)} records")

        st.subheader("📊 Data Preview")
        st.json(records[0] if records else {})

        query = st.text_input("💬 Ask a question about your file:")
        if query:
            with st.spinner("🔎 Asking GPT..."):
                gpt_ans = ask_gpt(query, records)
                st.markdown(f"**Answer:** {gpt_ans}")

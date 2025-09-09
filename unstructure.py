import os
import json
import streamlit as st
import numpy as np
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
from openai import OpenAI
import dateparser
import locale

# -----------------------------
# Init OpenAI
# -----------------------------
# Load API key from Streamlit secrets
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", None)

if not OPENAI_API_KEY:
    st.error("⚠️ Please set your OpenAI API key in .streamlit/secrets.toml")
else:
    client = OpenAI(api_key=OPENAI_API_KEY)


locale.setlocale(locale.LC_ALL, '')  # format numbers with commas

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
# Helpers
# -----------------------------
def flatten_json(y, prefix=""):
    out = {}
    def flatten(x, name=""):
        if isinstance(x, dict):
            for a in x:
                flatten(x[a], f"{name}{a}_")
        elif isinstance(x, list):
            i = 0
            for a in x:
                flatten(a, f"{name}{i}_")
                i += 1
        else:
            out[name[:-1]] = x
    flatten(y, prefix)
    return out

def extract_fields(records):
    fields = set()
    for r in records:
        flat = flatten_json(r)
        fields.update(flat.keys())
    return list(fields)

def match_field_to_query(query, fields):
    field_embeddings = [
        client.embeddings.create(input=f, model="text-embedding-3-small").data[0].embedding
        for f in fields
    ]
    q_emb = client.embeddings.create(input=query, model="text-embedding-3-small").data[0].embedding
    sims = [np.dot(q_emb, fe) / (np.linalg.norm(q_emb) * np.linalg.norm(fe)) for fe in field_embeddings]
    return fields[int(np.argmax(sims))]

def detect_date(value):
    if not value or not isinstance(value, str):
        return None
    return dateparser.parse(value)

# -----------------------------
# Main Answer Function
# -----------------------------
def answer_from_data(query, records):
    fields = extract_fields(records)
    if not fields:
        return None

    best_field = match_field_to_query(query, fields)
    query_date = dateparser.parse(query, settings={"PREFER_DAY_OF_MONTH": "first"})
    total = 0
    count = 0

    for r in records:
        flat = flatten_json(r)
        value = flat.get(best_field)

        try:
            value = float(value)
        except Exception:
            continue

        if query_date:
            date_candidates = [detect_date(v) for v in flat.values() if isinstance(v, str)]
            date_candidates = [d for d in date_candidates if d]
            if date_candidates:
                if any(d.month == query_date.month and d.year == query_date.year for d in date_candidates):
                    total += value
                    count += 1
            else:
                total += value
                count += 1
        else:
            total += value
            count += 1

    if count > 0:
        total_str = locale.format_string("%0.2f", total, grouping=True)
        month_year = f" for {query_date.strftime('%B %Y')}" if query_date else ""
        field_name = best_field.replace("_", " ").title()
        return f"✅ The total **{field_name}**{month_year} is **{total_str}**"
    return None

# -----------------------------
# GPT fallback
# -----------------------------
def ask_gpt(query, records, batch_size=50):
    """
    Generic GPT query processor for large datasets.
    Splits records into batches, extracts partial answers, then aggregates.
    """
    batch_answers = []
    total_records = len(records)

    # Step 1: Process in batches
    for i in range(0, total_records, batch_size):
        batch = records[i:i+batch_size]
        context = json.dumps(batch, indent=2)

        prompt = f"""
You are a data assistant. Answer the user query based ONLY on the given context. 
If you find relevant information, extract it clearly. 
If not found in this batch, just say "Not found".

User query:
{query}

Context (records {i+1}–{i+len(batch)} of {total_records}):
{context}

Answer (keep it concise and factual, do not hallucinate):
"""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        ans = response.choices[0].message.content.strip()
        batch_answers.append(ans)

    # Step 2: Aggregate all answers
    combined_context = "\n".join(batch_answers)

    final_prompt = f"""
The user query was:
{query}

Here are partial answers from multiple data batches:
{combined_context}

Now, combine them into ONE final clear answer. 
If multiple values exist, summarize the most relevant ones. 
Do not say 'Not found in this batch'. Just give the final best answer.
"""
    final_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": final_prompt}]
    )
    return final_response.choices[0].message.content

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="📂 Unstructured File Chatbot", layout="wide")
st.title("📂 Unstructured File Chatbot (XML / HTML / TXT)")

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
            with st.spinner("🔎 Finding answer..."):
                ans = answer_from_data(query, records)
                if ans:
                    st.markdown(f"**Answer:** {ans}")
                else:
                    st.markdown("🤔 No direct match found, asking GPT...")
                    gpt_ans = ask_gpt(query, records)
                    st.markdown(f"**Answer:** {gpt_ans}")

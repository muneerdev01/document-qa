import streamlit as st
from openai import OpenAI
import PyPDF2
import docx2txt
import pandas as pd

# ---------------- UI ----------------
st.set_page_config(page_title="ChatDoc AI", layout="wide")
st.title("💬 ChatDoc AI (Simple Production Version)")

# ---------------- API KEY ----------------
api_key = st.text_input("🔑 OpenAI API Key", type="password")

if not api_key:
    st.stop()

client = OpenAI(api_key=api_key)

# ---------------- SESSION STATE ----------------
if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "📄 Upload Document",
    type=["pdf", "txt", "md", "docx", "xlsx"]
)

# ---------------- TEXT EXTRACTION ----------------
def extract_text(file):
    ext = file.name.split(".")[-1].lower()

    if ext in ["txt", "md"]:
        return file.read().decode("utf-8")

    elif ext == "pdf":
        reader = PyPDF2.PdfReader(file)
        return "".join([p.extract_text() or "" for p in reader.pages])

    elif ext == "docx":
        return docx2txt.process(file)

    elif ext == "xlsx":
        df = pd.read_excel(file)
        return df.to_string(index=False)

    return ""

# ---------------- CHUNKING ----------------
def chunk_text(text, size=500):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

# ---------------- BUILD KNOWLEDGE BASE ----------------
if uploaded_file and not st.session_state.chunks:
    text = extract_text(uploaded_file)
    text = text[:15000]

    st.session_state.chunks = chunk_text(text)

    st.success("✅ Document loaded successfully!")

# ---------------- SIMPLE RETRIEVAL ----------------
def retrieve(query):
    query_words = set(query.lower().split())

    scored = []
    for chunk in st.session_state.chunks:
        score = len(query_words.intersection(set(chunk.lower().split())))
        scored.append((score, chunk))

    scored.sort(reverse=True, key=lambda x: x[0])

    return [c for _, c in scored[:3]]

# ---------------- GPT ANSWER ----------------
def ask_ai(query, context, history):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer using context only when relevant."
        }
    ]

    for h in history[-5:]:
        messages.append(h)

    messages.append({
        "role": "user",
        "content": f"""
Context:
{context}

Question:
{query}
"""
    })

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    return res.choices[0].message.content

# ---------------- CHAT UI ----------------
st.subheader("💬 Chat with Document")

query = st.text_input("Ask something")

if query and st.session_state.chunks:
    with st.spinner("Thinking..."):

        context = retrieve(query)

        answer = ask_ai(query, "\n\n".join(context), st.session_state.history)

        st.session_state.history.append({"role": "user", "content": query})
        st.session_state.history.append({"role": "assistant", "content": answer})

    st.markdown("### 🧠 Answer")
    st.write(answer)

# ---------------- HISTORY ----------------
st.markdown("### 🧾 Chat History")

for msg in st.session_state.history:
    st.write(f"**{msg['role']}**: {msg['content']}")

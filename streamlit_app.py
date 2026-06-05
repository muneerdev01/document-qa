import streamlit as st
from openai import OpenAI
import chromadb
import docx2txt
import PyPDF2
import pandas as pd
import uuid

# ---------------- CONFIG ----------------
st.set_page_config(page_title="ChatDoc AI", layout="wide")
st.title("💬 ChatDoc AI (Production RAG System)")

# ---------------- API KEY ----------------
api_key = st.text_input("🔑 OpenAI API Key", type="password")

if not api_key:
    st.warning("Enter API key to continue")
    st.stop()

client = OpenAI(api_key=api_key)

# ---------------- CHROMA DB ----------------
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("docs")

# ---------------- SESSION STATE ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "doc_id" not in st.session_state:
    st.session_state.doc_id = str(uuid.uuid4())

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "📄 Upload Document",
    type=["pdf", "txt", "md", "docx", "xlsx"]
)

# ---------------- TEXT EXTRACTION ----------------
def extract_text(file):
    ext = file.name.split(".")[-1].lower()

    try:
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

    except Exception as e:
        st.error(f"File error: {e}")
        return ""

    return ""

# ---------------- CHUNKING ----------------
def chunk_text(text, size=400):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

# ---------------- EMBEDDINGS ----------------
def get_embedding(text):
    return client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    ).data[0].embedding

# ---------------- STORE DOCS (FIXED) ----------------
def store_docs(chunks, doc_id):
    for i, chunk in enumerate(chunks):
        emb = get_embedding(chunk)

        collection.add(
            ids=[f"{doc_id}_{i}_{uuid.uuid4()}"],  # FIXED UNIQUE IDS
            embeddings=[emb],
            documents=[chunk],
            metadatas=[{"doc_id": doc_id}]
        )

# ---------------- RETRIEVE ----------------
def retrieve(query, k=3):
    q_emb = get_embedding(query)

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=k
    )

    return results["documents"][0] if results["documents"] else []

# ---------------- GPT RESPONSE ----------------
def ask_ai(query, context, history):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant. Use the provided context to answer accurately."
        }
    ]

    # memory (last 6 messages)
    for h in history[-6:]:
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

# ---------------- PROCESS DOCUMENT ----------------
if uploaded_file:
    text = extract_text(uploaded_file)

    if text:
        text = text[:15000]
        chunks = chunk_text(text)

        with st.spinner("📦 Indexing document..."):
            store_docs(chunks, st.session_state.doc_id)

        st.success("✅ Document ready!")

# ---------------- CHAT UI ----------------
st.subheader("💬 Chat with your document")

user_query = st.text_input("Ask anything")

if user_query:
    with st.spinner("Thinking..."):

        context = retrieve(user_query)

        answer = ask_ai(
            user_query,
            "\n\n".join(context),
            st.session_state.chat_history
        )

        st.session_state.chat_history.append(
            {"role": "user", "content": user_query}
        )
        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer}
        )

    st.markdown("### 🧠 Answer")
    st.write(answer)

# ---------------- HISTORY ----------------
st.markdown("### 🧾 Chat History")

for msg in st.session_state.chat_history:
    st.write(f"**{msg['role']}**: {msg['content']}")

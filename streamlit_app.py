import streamlit as st
from openai import OpenAI
import chromadb
import numpy as np
import docx2txt
import PyPDF2
import pandas as pd

# ---------------- UI ---------------- #
st.set_page_config(page_title="ChatDoc AI", layout="wide")
st.title("💬 ChatDoc AI (ChatGPT + RAG + Memory)")

# ---------------- API KEY ---------------- #
api_key = st.text_input("🔑 OpenAI API Key", type="password")

if not api_key:
    st.stop()

client = OpenAI(api_key=api_key)

# ---------------- CHROMA DB ---------------- #
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("docs")

# ---------------- SESSION MEMORY ---------------- #
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------- FILE UPLOAD ---------------- #
uploaded_file = st.file_uploader(
    "📄 Upload Document",
    type=["pdf", "txt", "md", "docx", "xlsx"]
)

# ---------------- EXTRACT TEXT ---------------- #
def extract_text(file):
    file_type = file.name.split(".")[-1].lower()

    if file_type in ["txt", "md"]:
        return file.read().decode("utf-8")

    elif file_type == "pdf":
        reader = PyPDF2.PdfReader(file)
        return "".join([p.extract_text() or "" for p in reader.pages])

    elif file_type == "docx":
        return docx2txt.process(file)

    elif file_type == "xlsx":
        df = pd.read_excel(file)
        return df.to_string(index=False)

    return ""

# ---------------- CHUNK ---------------- #
def chunk_text(text, size=400):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

# ---------------- EMBEDDING ---------------- #
def get_embedding(text):
    return client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    ).data[0].embedding

# ---------------- STORE IN CHROMA ---------------- #
def store_docs(chunks):
    for i, chunk in enumerate(chunks):
        emb = get_embedding(chunk)

        collection.add(
            ids=[str(i)],
            embeddings=[emb],
            documents=[chunk]
        )

# ---------------- RETRIEVE ---------------- #
def retrieve(query, k=3):
    q_emb = get_embedding(query)

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=k
    )

    return results["documents"][0]

# ---------------- GPT ANSWER ---------------- #
def ask_ai(query, context, history):
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant with memory and document understanding."}
    ]

    # memory
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

# ---------------- PROCESS DOCUMENT ---------------- #
if uploaded_file:

    text = extract_text(uploaded_file)
    text = text[:15000]

    chunks = chunk_text(text)

    with st.spinner("📦 Building knowledge base..."):
        store_docs(chunks)

    st.success("✅ Document indexed successfully!")

# ---------------- CHAT UI ---------------- #
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

        # save memory
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

    st.markdown("### 🧠 Answer")
    st.write(answer)

# ---------------- CHAT HISTORY ---------------- #
st.markdown("### 🧾 Chat History")

for msg in st.session_state.chat_history:
    st.write(f"**{msg['role']}**: {msg['content']}")

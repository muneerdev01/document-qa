import streamlit as st
from openai import OpenAI
import faiss
import numpy as np
import pandas as pd
import docx2txt
import PyPDF2

# ---------------- UI ---------------- #
st.set_page_config(page_title="Doc AI Assistant", layout="wide")
st.title("📄 Smart Document Q&A System (RAG + Multi-File Support)")

# ---------------- API KEY ---------------- #
api_key = st.text_input("🔑 Enter OpenAI API Key", type="password")

if not api_key:
    st.info("Please enter your OpenAI API key to continue.")
    st.stop()

client = OpenAI(api_key=api_key)

# ---------------- HELP ---------------- #
with st.expander("🔑 How to get OpenAI API Key?"):
    st.markdown("""
1. Go to https://platform.openai.com/
2. Sign up / Login
3. Open API Keys section
4. Click "Create new secret key"
5. Copy and paste here

⚠️ Never share your API key with anyone.
""")

guide_text = """
OpenAI API Key Guide

Step 1: https://platform.openai.com/
Step 2: Sign up / Login
Step 3: Go to API Keys
Step 4: Create new secret key
Step 5: Paste in app

Note:
- Keep key private
- Usage may be paid
"""

st.download_button(
    "📄 Download API Key Guide",
    guide_text,
    "api_key_guide.txt",
    "text/plain"
)

# ---------------- FILE UPLOAD ---------------- #
uploaded_file = st.file_uploader(
    "📤 Upload your document",
    type=["pdf", "txt", "md", "docx", "xlsx"]
)

# ---------------- TEXT EXTRACTION ---------------- #
def extract_text(file):
    file_type = file.name.split(".")[-1].lower()

    if file_type in ["txt", "md"]:
        return file.read().decode("utf-8")

    elif file_type == "pdf":
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text

    elif file_type == "docx":
        return docx2txt.process(file)

    elif file_type == "xlsx":
        df = pd.read_excel(file)
        return df.to_string(index=False)

    return "Unsupported file type"

# ---------------- RAG FUNCTIONS ---------------- #
def chunk_text(text, chunk_size=400):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]


def get_embedding(text):
    return client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    ).data[0].embedding


def build_vector_store(chunks):
    embeddings = [get_embedding(c) for c in chunks]
    dim = len(embeddings[0])

    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))

    return index, chunks


def retrieve(query, index, chunks, k=3):
    q_emb = np.array([get_embedding(query)]).astype("float32")
    _, indices = index.search(q_emb, k)
    return [chunks[i] for i in indices[0]]


def generate_answer(question, context):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a professional document analyst. Answer clearly and structured."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{question}"
            }
        ]
    )
    return response.choices[0].message.content

# ---------------- MAIN ---------------- #
if uploaded_file:

    document_text = extract_text(uploaded_file)

    # safety limit
    document_text = document_text[:12000]

    st.success("✅ Document Loaded Successfully")

    chunks = chunk_text(document_text)

    with st.spinner("🔄 Building knowledge base..."):
        index, chunks = build_vector_store(chunks)

    st.success("✅ RAG System Ready!")

    question = st.text_area("❓ Ask a question about your document")

    if question:
        with st.spinner("🧠 Thinking..."):

            relevant_chunks = retrieve(question, index, chunks)
            context = "\n\n".join(relevant_chunks)

            answer = generate_answer(question, context)

        st.subheader("🧠 Answer")
        st.write(answer)

        st.subheader("📌 Retrieved Context")
        st.write(context)

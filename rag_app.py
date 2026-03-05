#  优化1+2
import os
import tempfile
import time
import hashlib
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# =========================
# OpenAI API Key
# =========================
os.environ["OPENAI_API_KEY"] = "" 

# =========================
# LLM & Embeddings
# =========================
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0.4
)

# =========================
# System Prompt (Priority 1)
# =========================
SYSTEM_PROMPT = """
You are a financial analysis assistant answering questions strictly based on the provided 10-K documents.

Rules:
- Base your answer ONLY on the retrieved document excerpts.
- If the documents do not contain enough information to answer the question, explicitly say:
  "The provided 10-K documents do not contain sufficient information to answer this question."
- Do NOT use external knowledge or assumptions.
- If comparing companies or years, ensure the comparison is supported by the documents.
"""

PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
{context}

Question:
{question}

Answer:
"""
)

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="📄 10-K RAG Chatbot", layout="wide")
st.title("📊 Chat with 10-K Reports (Amazon / Google / Microsoft)")

uploaded_files = st.file_uploader(
    "Upload 10-K PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

# =========================
# Helper: file hash
# =========================
def get_files_hash(files):
    hasher = hashlib.md5()
    for f in sorted(files, key=lambda x: x.name):
        hasher.update(f.name.encode())
        hasher.update(f.getvalue())
    return hasher.hexdigest()

# =========================
# Process PDFs
# =========================
if uploaded_files:
    files_hash = get_files_hash(uploaded_files)

    if st.session_state.get("files_hash") != files_hash:
        with st.spinner("Processing 10-K PDFs..."):
            documents = []

            with tempfile.TemporaryDirectory() as temp_dir:
                for file in uploaded_files:
                    file_path = os.path.join(temp_dir, file.name)
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())

                    loader = PyPDFLoader(file_path)
                    documents.extend(loader.load())

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1200,
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(documents)

            vector_store = FAISS.from_documents(chunks, embeddings)

            st.session_state.vector_store = vector_store
            st.session_state.files_hash = files_hash

        st.success("✅ 10-K documents indexed successfully!")

# =========================
# Chat history
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =========================
# User input
# =========================
user_input = st.chat_input("Ask a question about the 10-K reports...")

if user_input and "vector_store" in st.session_state:
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    # =========================
    # Priority 2: Multi-Query Retriever
    # =========================
    base_retriever = st.session_state.vector_store.as_retriever(
        search_kwargs={"k": 6}
    )

    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=multi_query_retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )

    with st.spinner("Thinking..."):
        response = qa_chain.invoke({
            "query": user_input
        })

    answer = response["result"]
    sources = response.get("source_documents", [])

    with st.chat_message("assistant"):
        placeholder = st.empty()
        output = ""
        for token in answer.split():
            output += token + " "
            placeholder.markdown(output)
            time.sleep(0.02)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

    if sources:
        with st.expander("📚 Retrieved Context"):
            for i, doc in enumerate(sources):
                st.markdown(f"**Chunk {i+1}**")
                st.markdown(doc.page_content)
                st.markdown(
                    f"_Source: Page {doc.metadata.get('page', 'unknown')}_"
                )
                st.markdown("---")

elif user_input:
    st.warning("Please upload 10-K PDFs first.")

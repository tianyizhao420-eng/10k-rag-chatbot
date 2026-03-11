import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import tempfile
import time

# -----------------------------
# Config
# -----------------------------
persona = """
You are a helpful assistant that answers questions based only on the provided documents.
Use the retrieved context to answer the user's question as accurately as possible.
If the answer is not in the context, say: "I don't have enough information to answer this question."
When possible, cite specific details from the retrieved context.
If the question clearly refers to a specific company, do not answer using another company's information.
"""

qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
{persona}

Context:
{context}

Question:
{question}

Answer:
""".strip().replace("{persona}", persona)
)

OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_LLM_MODEL = "mistral"

llm = OllamaLLM(model=OLLAMA_LLM_MODEL, temperature=0.3)
embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)

st.set_page_config(page_title="Chat with Your PDFs (Ollama)")
st.title("📄💬 Chat with Your PDFs (Ollama)")

# -----------------------------
# Helpers
# -----------------------------
def detect_company_from_filename(filename: str) -> str:
    name = filename.lower()
    if "amazon" in name or "amzn" in name:
        return "amazon"
    if "microsoft" in name or "msft" in name:
        return "microsoft"
    if "alphabet" in name or "google" in name or "goog" in name:
        return "alphabet"
    return "unknown"


def detect_company_from_query(query: str) -> str | None:
    q = query.lower()
    if "amazon" in q or "amzn" in q:
        return "amazon"
    if "microsoft" in q or "msft" in q:
        return "microsoft"
    if "alphabet" in q or "google" in q or "goog" in q:
        return "alphabet"
    return None


def build_vector_store(uploaded_files):
    documents = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for file in uploaded_files:
            temp_file_path = os.path.join(temp_dir, file.name)
            with open(temp_file_path, "wb") as f:
                f.write(file.getbuffer())

            loader = PyPDFLoader(temp_file_path)
            loaded_docs = loader.load()

            company = detect_company_from_filename(file.name)

            for doc in loaded_docs:
                doc.metadata["source_file"] = file.name
                doc.metadata["company"] = company

            documents.extend(loaded_docs)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    split_docs = text_splitter.split_documents(documents)

    return FAISS.from_documents(split_docs, embeddings), split_docs


# -----------------------------
# Session init
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "processed_files" not in st.session_state:
    st.session_state.processed_files = None

if "all_docs" not in st.session_state:
    st.session_state.all_docs = []

# -----------------------------
# Upload
# -----------------------------
uploaded_files = st.file_uploader(
    "Upload PDFs",
    accept_multiple_files=True,
    type=["pdf"]
)

if uploaded_files:
    current_files = sorted([file.name for file in uploaded_files])

    if st.session_state.processed_files != current_files:
        with st.spinner("Processing your PDFs..."):
            vector_store, all_docs = build_vector_store(uploaded_files)
            st.session_state.vector_store = vector_store
            st.session_state.all_docs = all_docs
            st.session_state.processed_files = current_files
            st.session_state.messages = []

        st.success("✅ PDFs uploaded and processed! You can now start chatting.")

    st.write("Processed files:")
    for f in current_files:
        st.write(f"- {f} ({detect_company_from_filename(f)})")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    user_input = st.chat_input("Ask a question about your PDFs...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        target_company = detect_company_from_query(user_input)

        # Retrieve docs
        with st.spinner("Thinking..."):
            retrieved_docs = []

            if target_company:
                # Manual metadata filtering for FAISS
                company_docs = [
                    doc for doc in st.session_state.all_docs
                    if doc.metadata.get("company") == target_company
                ]

                if company_docs:
                    temp_company_store = FAISS.from_documents(company_docs, embeddings)
                    retrieved_docs = temp_company_store.max_marginal_relevance_search(
                        user_input,
                        k=4,
                        fetch_k=20,
                        lambda_mult=0.7
                    )
                else:
                    retrieved_docs = st.session_state.vector_store.max_marginal_relevance_search(
                        user_input,
                        k=4,
                        fetch_k=20,
                        lambda_mult=0.7
                    )
            else:
                retrieved_docs = st.session_state.vector_store.max_marginal_relevance_search(
                    user_input,
                    k=4,
                    fetch_k=20,
                    lambda_mult=0.7
                )

            # Build a retriever from the full store for QA chain
            # We override the context effectively through retrieval behavior above by using a filtered temp store when needed.
            if target_company:
                company_docs = [
                    doc for doc in st.session_state.all_docs
                    if doc.metadata.get("company") == target_company
                ]
                if company_docs:
                    qa_store = FAISS.from_documents(company_docs, embeddings)
                else:
                    qa_store = st.session_state.vector_store
            else:
                qa_store = st.session_state.vector_store

            retriever = qa_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.7}
            )

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                chain_type="stuff",
                return_source_documents=True,
                chain_type_kwargs={
                    "prompt": qa_prompt
                }
            )

            response = qa_chain.invoke({"query": user_input})

            if "source_documents" in response:
                with st.expander("View Retrieved Chunks (Context)"):
                    for i, doc in enumerate(response["source_documents"]):
                        st.markdown(f"**Chunk {i+1}**")
                        st.markdown(f"**Source File:** {doc.metadata.get('source_file', 'unknown')}")
                        st.markdown(f"**Company:** {doc.metadata.get('company', 'unknown')}")
                        st.markdown(f"**Page:** {doc.metadata.get('page', 'unknown')}")
                        st.markdown(f"**Content:** {doc.page_content}")
                        st.markdown("---")

            response_text = response["result"]

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for chunk in response_text.split():
                full_response += chunk + " "
                message_placeholder.markdown(full_response)
                time.sleep(0.03)

        st.session_state.messages.append({"role": "assistant", "content": response_text})

else:
    st.info("Please upload PDF files to begin.")
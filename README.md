# 📊 10-K RAG Chatbot

An interactive **Retrieval-Augmented Generation (RAG)** chatbot for analyzing **10-K financial reports** from major technology companies such as **Amazon, Microsoft, and Alphabet (Google)**.

Users can upload multiple 10-K PDF files and ask questions about financial performance, business risks, and company operations.

The system retrieves relevant sections from the documents and generates answers grounded strictly in the retrieved content.

---

# 🧠 Project Goal

The goal of this project is to build a **RAG-based financial document QA system** and compare the performance of:

- **Weak LLM (local model)**
- **Strong LLM (API-based model)**

We evaluate the models based on:

- Answer accuracy  
- Hallucination behavior  
- Retrieval quality  
- Response latency  

---

# 🏗️ System Architecture

The chatbot follows a standard **Retrieval-Augmented Generation (RAG)** pipeline:
PDF Documents (10-K Reports)
↓
Document Loader (PyPDFLoader)
↓
Text Chunking
↓
Embedding Model
↓
Vector Database (FAISS)
↓
Retriever
↓
Large Language Model (LLM)
↓
Answer grounded in retrieved document content
---

# 🤖 Weak vs Strong LLM

This project implements two different RAG pipelines.

| Component | Weak LLM | Strong LLM |
|---|---|---|
| LLM | Mistral (local) | OpenAI GPT |
| Embedding Model | nomic-embed-text | OpenAI embeddings |
| Deployment | Local via Ollama | Cloud API |
| Vector DB | FAISS | FAISS |

### Weak LLM

Runs locally using **Ollama** with the **Mistral** model.

Advantages:
- Free and runs locally
- No API usage

Limitations:
- Weaker reasoning
- More prone to hallucination

### Strong LLM

Uses **OpenAI GPT models** with OpenAI embeddings.

Advantages:
- Better reasoning
- More accurate responses

Limitations:
- Requires API key
- Higher cost

---

# 📂 Repository Structure
10k-rag-chatbot
│
├── rag_weakllm.py      # Weak LLM RAG implementation (Ollama + Mistral)
├── rag_strongllm.py    # Strong LLM RAG implementation (OpenAI)
│
├── sample_strongllm.md # Example questions
│
└── README.md
---

# 📑 Dataset

This project uses **10-K Annual Reports** from major companies:

- Amazon
- Microsoft
- Alphabet (Google)

These reports contain detailed information about:

- Business segments
- Risk factors
- Financial performance
- Company strategy

---

# 🚀 Running the Weak LLM Version

### 1. Install Ollama

Install Ollama from:

https://ollama.com/

### 2. Pull Required Models
```ollama pull mistral```
```ollama pull nomic-embed-text```
### 3. Install Python Dependencies
```pip install -r requirements.txt```
### 4. Run the App
```streamlit run rag_weakllm.py```

---

# 🚀 Running the Strong LLM Version

### 1. Install Dependencies
```pip install -r requirements.txt```
### 2. Set OpenAI API Key

Mac / Linux
```export OPENAI_API_KEY=“your_api_key”```
Windows
```set OPENAI_API_KEY=“your_api_key”```
### 3. Run the Application
```streamlit run rag_strongllm.py```

---

# 💬 Example Questions

Users can ask questions such as:

- What are Amazon's main revenue sources?
- What risk factors are mentioned in Microsoft's 10-K?
- How does Alphabet generate advertising revenue?
- Compare AWS and Azure businesses.

---

# 📊 Evaluation Focus

We compare Weak and Strong LLM performance based on:

| Metric | Description |
|---|---|
| Accuracy | Correctness of generated answers |
| Hallucination | Whether the model invents unsupported information |
| Retrieval Quality | Relevance of retrieved document chunks |
| Latency | Response time |

---

# 🧑‍💻 Authors

Course Project: **10-K Financial Document QA System using RAG**

Team contributions include:

- Weak LLM RAG implementation
- Strong LLM RAG implementation
- Retrieval evaluation and analysis

---

# 🔎 Key Takeaways

- RAG improves answer reliability by grounding responses in source documents.
- Strong LLMs provide better reasoning and accuracy.
- Weak LLMs can still perform well for factual retrieval-based questions.
- Proper retrieval configuration helps reduce hallucination.

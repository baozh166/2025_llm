# 🧠 2025_LLM — Lightweight Local RAG with Qdrant

This repository demonstrates a **Retrieval-Augmented Generation (RAG)** pipeline using a local **Qdrant vector database** to index and query text documents.  
The example dataset is a **medical QA corpus**, and the code shows how to embed, store, and retrieve context vectors to power a simple RAG workflow.

---

## 📦 Features
- ✅ **Automated Qdrant setup** (runs a Docker container if not running)  
- ✅ **Vector ingestion** with local embeddings (`jinaai/jina-embeddings-v2-small-en`)  
- ✅ **Fast retrieval** from Qdrant for semantic (vector) search  
- ✅ **Simple RAG entrypoint** (`main.py` + `run_test.sh`)  

---

## 🧩 Architecture Overview

          +----------------------+
          |   medquad.csv Data   |
          +----------+-----------+
                     |
                     v
          +----------------------+
          |  ingest_vec.py       |
          |  - Load + embed data |
          |  - Start Qdrant DB   |
          +----------+-----------+
                     |
                     v
          +----------------------+
          |   Qdrant Vector DB   |
          |  (medicalQA-rag1020) |
          +----------+-----------+
                     |
                     v
          +----------------------+
          |   main.py (RAG)      |
          |  - Query embeddings  |
          |  - Retrieve matches  |
          +----------+-----------+
                     |
                     v
          +----------------------+
          |  rag_vec.rag()       |
          |  - Retrieve context  |
          |  - Generate response |
          +----------------------+


---

## ⚙️ Requirements

### 1. System
- Python ≥ 3.9  
- Docker (for running Qdrant locally)  

### 2. Python packages
Install dependencies:
```bash
pip install pandas qdrant-client jina

## 🧠 How It Works

1. **Data Ingestion (`ingest_vec.py`)**
   - Starts a Qdrant container automatically.  
   - Creates a collection if not already present.  
   - Embeds and inserts documents as vectors with metadata.

2. **Query Phase (`main.py`)**
   - Calls `rag_vec.rag(query)` (your retrieval + generation logic).  
   - Finds the most similar documents to the query vector.

3. **Testing (`run_test.sh`)**
   - Example single query run for smoke testing.

## 🚀 Quick Start

### Step 1. Prepare Data

Ensure a CSV file (e.g., `data/medquad.csv`) exists with at least these columns:

```text
source, focus_area, answer


### Step 2. Ingest Data

Run the ingestion script to spin up Qdrant and populate it with vectorized data:

```bash
python3 ingest_vec.py
This script will:

Pull the Qdrant Docker image (if not already available).

Run Qdrant in a local container.

Read the dataset (data/medquad.csv).

Embed the answers using the Jina embeddings model.

Upload vectors and metadata to the Qdrant collection (medicalQA-rag1020).

### Step 3. Run a Query

Once Qdrant is running and indexed, test the RAG retrieval with:

```bash
bash run_test.sh

or directly

```bash
python3 main.py --query "what is the reason to develop a lung cancer?"
This triggers the RAG process (rag_vec.rag()), which performs a semantic search against Qdrant and returns the most relevant entries.



# RAG Agentic AI 

A Retrieval-Augmented Generation (RAG) chatbot that answers questions from the *Agentic AI: An Executive's Guide* eBook.

The main idea is simple: instead of letting the LLM answer from its general knowledge, the application first searches the eBook for relevant sections and then uses those sections to generate the answer. If the retrieved content isn't relevant enough, the system refuses to answer.

## Features

- **Document-grounded answers** — The LLM receives only the relevant chunks retrieved from the eBook.
- **LangGraph workflow** — The question goes through retrieval first and is then routed either to answer generation or to a refusal step.
- **Local LLM and embeddings** — Uses Ollama with `llama3.2:3b` and Hugging Face `all-MiniLM-L6-v2`, so no LLM API key or API quota is required.
- **Streamlit interface** — Provides a simple chat interface and shows the retrieved chunks and confidence score alongside the answer.
- **Out-of-scope protection** — Questions that don't have enough relevant information in the eBook are rejected instead of being answered from outside knowledge.

## How it works

The eBook is loaded, split into smaller chunks, converted into embeddings, and stored in Pinecone.

When a user asks a question:

1. The question is converted into an embedding.
2. Pinecone retrieves the most relevant chunks.
3. The best retrieval score is compared with the configured relevance threshold.
4. If the score is too low, the system refuses to answer.
5. If the score is high enough, the retrieved context is passed to the local Ollama model.
6. The final response, retrieved chunks, and confidence score are returned to the UI.

### Architecture

```text
PDF
 │
 ▼
Load & chunk (800 / 150)
 │
 ▼
MiniLM embeddings
 │
 ▼
Pinecone vector index
 │
 │
 ▼
User question
 │
 ▼
Retrieve top-K chunks
 │
 ▼
Relevance check
 ┌─────────────────────┐
 │                     │
 ▼                     ▼
Relevant            Not relevant
 │                     │
 ▼                     ▼
Ollama              Refuse
 │
 ▼
Answer + confidence + retrieved chunks
```

## Tech stack

| Component | Technology |
|---|---|
| Language | Python |
| Orchestration | LangGraph |
| LLM | Ollama — `llama3.2:3b` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Embedding size | 384 dimensions |
| Vector database | Pinecone (serverless, cosine similarity) |
| UI | Streamlit |

## Project structure

```text
.
├── data/
│   └── Ebook-Agentic-AI.pdf
├── src/
│   ├── config.py          # Models, chunking and thresholds
│   ├── ingestion.py       # Load, chunk, embed and upload to Pinecone
│   └── graph.py           # LangGraph workflow
├── app.py                 # Streamlit chat interface
├── tests_sample_queries.py
├── requirements.txt
└── .env
```

## Setup

### Prerequisites

You need:

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- A free [Pinecone](https://www.pinecone.io) account

### 1. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

### 2. Install the dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the Ollama model

```bash
ollama pull llama3.2:3b
```

### 4. Configure Pinecone

Create a `.env` file in the project root:

```env
PINECONE_API_KEY=your_key_here
PINECONE_INDEX_NAME=agentic-ai-index-hf
```

### 5. Add the eBook

Place the source document here:

```text
data/Ebook-Agentic-AI.pdf
```

## Running the project

First, ingest the eBook and create/update the Pinecone index:

```bash
python -m src.ingestion
```

Then start the Streamlit application:

```bash
streamlit run streamlit_app.py
```

The first question can take longer because Ollama needs to load the model into memory. On a CPU-only machine, an answer can take roughly 10–45 seconds.

You can also run the sample questions directly from the terminal:

```bash
python tests_sample_queries.py
```

## Configuration

The main settings are in `src/config.py`.

| Setting | Current value | Purpose |
|---|---|---|
| `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `EMBED_DIM` | `384` | Pinecone index dimension |
| `LLM_MODEL` | `llama3.2:3b` | Local Ollama model |
| `CHUNK_SIZE` | `800` | Chunk size |
| `CHUNK_OVERLAP` | `150` | Overlap between chunks |
| `TOP_K` | `3` | Number of chunks retrieved per question |
| `RELEVANCE_THRESHOLD` | `0.4` | Minimum retrieval score required before generation |

 


## Screenshots

### Agentic AI Definition

<img width="959" height="539" alt="agentic-ai-definition" src="https://github.com/user-attachments/assets/1c32f364-48df-43fc-b081-682747d610f0" />

### Agentic Architecture

<img width="958" height="536" alt="agentic-architecture-question" src="https://github.com/user-attachments/assets/ba2f4941-ccbe-4a05-b50c-96288f1a5007" />

### Industry Use Cases

<img width="959" height="532" alt="industry-use-cases" src="https://github.com/user-attachments/assets/25392f37-cbee-4e30-b13c-38fb323c6bee" />

### Agentic AI and Traditional AI

<img width="958" height="535" alt="agentic-ai-and-traditional-ai" src="https://github.com/user-attachments/assets/ed374a97-fd5f-42e7-8e65-215e4beda097" />

### Agentic AI Challenges

<img width="959" height="533" alt="agentic-ai-challenges" src="https://github.com/user-attachments/assets/8a548803-d249-44a8-aef4-03294b092943" />

### Out-of-Scope Question and Refusal

<img width="959" height="533" alt="out-of-scope-refusal" src="https://github.com/user-attachments/assets/33f721ba-69ae-4465-b073-3b55622ae89f" />









 

# RAG Project

A local Retrieval-Augmented Generation (RAG) system. Ask questions in natural
language against your own documents (PDF, TXT, HTML) and get grounded answers —
the model only uses what's in the documents, never invents facts.

## How it works

```
Documents (PDF / TXT / HTML)
        │
        ▼
   [1] INGEST (ingest.py)
   Read → chunk → embed (BGE-M3) → store in Qdrant
        │
        ▼
   [2] QUERY (main.py — FastAPI server)
   Question → hybrid search (dense + sparse, RRF fusion)
            → cross-encoder rerank → Qwen 2.5 7B → streaming answer
        │
        ▼
   [3] CHAT (chat.py — CLI client)
   Streams the answer token by token
```

---

## Prerequisites

Install these before anything else.

| Requirement | Notes |
|---|---|
| Python 3.10+ | `python3 --version` to check |
| NVIDIA GPU + CUDA | BGE-M3 and the reranker require a GPU |
| [Ollama](https://ollama.com) | Runs the LLM locally |
| [Qdrant](https://qdrant.tech/documentation/quick-start/) | Vector database, runs as a Docker container |

### Pull the LLM

```bash
ollama pull qwen2.5:7b
```

### Start Qdrant (Docker)

```bash
docker run -d -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

---

## Setup

### 1. Clone the repo

```bash
git clone <your-repo-url>
cd rag-project
```

### 2. Create a virtual environment

A virtual environment is an isolated Python installation for this project — it
keeps dependencies separate from the rest of your system.

```bash
python3 -m venv venv
```

This creates a `venv/` folder. You need to **activate** it every time you open
a new terminal before running anything:

```bash
source venv/bin/activate      # Linux / macOS / WSL
# venv\Scripts\activate       # Windows CMD/PowerShell
```

Your prompt will show `(venv)` when it's active. To deactivate: `deactivate`.

### 3. Install PyTorch with CUDA

Do this before the other requirements, replacing `cu121` with your CUDA version
(`cu118`, `cu121`, `cu124`, `cu128` — check with `nvcc --version`):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

### 4. Install the remaining dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Step 1 — Add your documents

Place PDF, TXT, or HTML files in the `docs/` folder.

### Step 2 — Ingest documents (run once, or after adding new files)

```bash
python ingest.py
```

This reads all files in `docs/`, chunks them, embeds them with BGE-M3, and
stores everything in Qdrant. Takes a few minutes the first time.

### Step 3 — Start the API server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Leave this running in its own terminal.

### Step 4 — Chat

Open a new terminal, activate the venv, then:

```bash
python chat.py
```

Type your question and press Enter. Type `exit` to quit.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/chat` | Single response (JSON) |
| `POST` | `/chat/stream` | Streaming response (NDJSON) |
| `GET` | `/debug?q=...` | Show retrieved chunks and scores |

Example with curl:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the maintenance schedule?"}'
```

---

## Project Structure

```
rag-project/
├── main.py               # FastAPI server — retrieval + LLM
├── ingest.py             # Document ingestion pipeline
├── chat.py               # CLI chat client
├── docs/
│   └── extract.py        # Utility for document extraction
├── requirements.txt
├── .gitignore
└── README.md
```

**Note:** The `docs/` folder is where you place your documents.
Data files (PDF, TXT, HTML) are excluded from git — only scripts are tracked.

---

## Configuration

Key constants at the top of each file:

**`main.py`**
- `HYBRID_LIMIT = 40` — candidates retrieved per search leg
- `RERANK_TOP_K = 8` — chunks sent to the LLM after reranking

**`ingest.py`**
- `PARENT_SIZE = 1500` — characters per parent chunk (sent to LLM)
- `CHILD_SIZE = 400` — characters per child chunk (used for search)
- `COLLECTION = "documents"` — Qdrant collection name

**`chat.py`**
- `BASE_URL = "http://localhost:8000"` — API server address

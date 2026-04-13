# RAG Project

A local Retrieval-Augmented Generation (RAG) system with an agentic layer.
Ask questions in natural language against your own documents (PDF, TXT, HTML)
and get grounded answers — the model only uses what's in the documents, never
invents facts. The agent can also perform multi-step tasks, do calculations,
and remember context across a conversation.

## How it works

### 1. Ingestion

```
Documents (PDF / TXT / HTML)
        │
        ▼
   [ingest.py]
   Read → parent-child chunking → embed with BGE-M3 (dense + sparse)
        → store in Qdrant (local vector database)
```

Each document is split into large *parent* chunks (1500 chars, sent to the LLM)
and small *child* chunks (400 chars, used for precise search). Both a dense
semantic vector and a sparse lexical vector are stored per child chunk.

### 2. The agentic query loop

```
User question
      │
      ▼
 Skill Router — keyword match → picks a skill (e.g. "maintenance", "general")
      │
      ▼
 AgentWorkflow (LlamaIndex ReActAgent)
      │
      ├─ think: what do I need to answer this?
      │
      ├─ call tool ──► search_documents
      │                  │
      │                  ├─ hybrid search on Qdrant (dense + sparse, RRF fusion)
      │                  ├─ deduplicate by parent chunk
      │                  └─ cross-encoder rerank → top 8 chunks returned
      │
      ├─ call tool ──► calculate / list_documents / get_current_date
      │                  (chained if needed to answer the question)
      │
      ├─ observe results → think again
      │
      └─ generate final answer (streamed token by token)
```

The key difference from a plain RAG system is the **think–act–observe loop**.
Instead of always running the same retrieve-then-answer pipeline, the agent
decides *which tools to call*, *in what order*, and *whether it needs more
information* before answering. A question like "calculate 15% of the budget
mentioned in the maintenance doc" will make the agent call `search_documents`
first, extract the number, then call `calculate` — without any hardcoded logic.

### 3. Skills

Skills are JSON files in `skills/` that control agent behaviour per domain:

```
skills/
  general.json      — fallback for generic document questions
  maintenance.json  — technical/maintenance queries (numbered steps, precise)
```

Each skill defines:
- **keywords** — trigger words that route the question to this skill
- **system_prompt** — the persona and rules the agent follows
- **tools** — which tools are available in this context

Add a new skill by dropping a `.json` file in `skills/` and restarting the server.

### 4. Conversation memory

Each `chat.py` session gets a UUID. The server keeps a `ChatMemoryBuffer` per
session (sliding window of 4096 tokens). Follow-up questions like
*"puoi spiegarlo meglio?"* work without repeating context. Type `reset` in the
CLI to clear memory and start fresh.

### 5. Available tools

| Tool | What it does |
|---|---|
| `search_documents` | Hybrid semantic + lexical search with reranking |
| `list_documents` | Lists all files currently indexed in Qdrant |
| `calculate` | Evaluates math expressions, percentages, square roots |
| `get_current_date` | Returns today's date |

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
| `GET` | `/skills` | List loaded skills and their tools |
| `GET` | `/debug?q=...` | Show raw retrieved chunks and reranking scores |
| `DELETE` | `/session/{id}` | Clear a session's conversation memory |

Both `/chat` and `/chat/stream` accept a `session_id` field (optional, defaults
to `"default"`). `chat.py` generates a UUID automatically per run.

Example with curl:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Quali sono i passi per il reset?", "session_id": "test"}'
```

---

## Project Structure

```
rag-project/
├── main.py               # FastAPI server — agent loop, tools, skills, retrieval
├── ingest.py             # Document ingestion pipeline
├── chat.py               # CLI chat client (streaming, session memory)
├── skills/
│   ├── general.json      # Fallback skill
│   └── maintenance.json  # Maintenance/technical skill
├── docs/
│   └── extract.py        # Utility for document extraction
├── agent_upgrade_plan.md # Notes on future agent improvements
├── requirements.txt
├── .gitignore
└── README.md
```

**Note:** The `docs/` folder is where you place your documents.
Data files (PDF, TXT, HTML) are excluded from git — only scripts are tracked.
Skill definitions in `skills/` are tracked and should be committed.

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

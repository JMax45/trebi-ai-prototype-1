from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from llama_index.llms.ollama import Ollama
from llama_index.core.agent import AgentWorkflow
from llama_index.core.agent.workflow.base_agent import AgentStream
from llama_index.core.tools import FunctionTool
from llama_index.core.memory import ChatMemoryBuffer
from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, FusionQuery, Fusion, SparseVector
import json
import pathlib
import sympy
import traceback
from datetime import date

# ── Config ────────────────────────────────────────────────────────────────────
COLLECTION         = "documents"
HYBRID_LIMIT       = 40    # candidates per search leg (dense + sparse)
RERANK_TOP_K       = 8     # parent chunks sent to LLM after reranking
MEMORY_TOKEN_LIMIT = 4096  # sliding-window conversation memory per session

# ── Models ────────────────────────────────────────────────────────────────────
print("Loading BGE-M3 …")
embed_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

print("Loading reranker …")
reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)

print("Loading LLM …")
llm = Ollama(model="qwen2.5:7b", request_timeout=120.0)

# ── Qdrant ────────────────────────────────────────────────────────────────────
qdrant = QdrantClient("localhost", port=6333)

app = FastAPI()

class Query(BaseModel):
    question: str
    session_id: str = "default"

# ── Retrieval ─────────────────────────────────────────────────────────────────
def encode_query(question: str):
    out = embed_model.encode(
        [question],
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    dense = out["dense_vecs"][0].tolist()
    sw    = out["lexical_weights"][0]
    return dense, [int(k) for k in sw], [float(v) for v in sw.values()]

def retrieve_context(question: str) -> list[dict]:
    dense, s_idx, s_val = encode_query(question)

    hits = qdrant.query_points(
        collection_name=COLLECTION,
        prefetch=[
            Prefetch(query=dense, using="dense", limit=HYBRID_LIMIT),
            Prefetch(
                query=SparseVector(indices=s_idx, values=s_val),
                using="sparse",
                limit=HYBRID_LIMIT,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=HYBRID_LIMIT,
        with_payload=True,
    ).points

    seen: dict[str, object] = {}
    for hit in hits:
        pid = hit.payload["parent_id"]
        if pid not in seen:
            seen[pid] = hit
    unique = list(seen.values())

    if not unique:
        return []

    texts  = [h.payload["parent_text"] for h in unique]
    scores = reranker.compute_score(
        [[question, t] for t in texts],
        normalize=True,
    )
    if isinstance(scores, float):
        scores = [scores]

    ranked = sorted(zip(scores, unique), key=lambda x: x[0], reverse=True)
    return [
        {"score": s, "text": h.payload["parent_text"], "file": h.payload["file_name"]}
        for s, h in ranked[:RERANK_TOP_K]
    ]

# ── Tools ─────────────────────────────────────────────────────────────────────
def search_documents(query: str) -> str:
    """Cerca informazioni nei documenti interni aziendali. Usare sempre questo strumento prima di rispondere a domande sui contenuti."""
    contexts = retrieve_context(query)
    if not contexts:
        return "Nessun documento rilevante trovato per questa query."
    parts = [
        f"[Documento {i+1} — {c['file']} | score={c['score']:.3f}]\n{c['text']}"
        for i, c in enumerate(contexts)
    ]
    return "\n\n---\n\n".join(parts)

def list_documents() -> str:
    """Elenca tutti i documenti disponibili nella base di conoscenza."""
    try:
        result = qdrant.scroll(
            collection_name=COLLECTION,
            with_payload=["file_name"],
            limit=1000,
        )
        file_names = sorted({p.payload["file_name"] for p in result[0]})
        if not file_names:
            return "Nessun documento indicizzato."
        return "Documenti disponibili:\n" + "\n".join(f"- {f}" for f in file_names)
    except Exception as e:
        return f"Errore nel recupero della lista documenti: {e}"

def calculate(expression: str) -> str:
    """Calcola un'espressione matematica. Esempi: '2 * (3 + 4)', 'sqrt(144)', '15% * 200'."""
    try:
        result = sympy.sympify(expression, evaluate=True)
        return str(result.evalf())
    except Exception as e:
        return f"Errore nel calcolo: {e}"

def get_current_date() -> str:
    """Restituisce la data odierna."""
    return date.today().strftime("%d/%m/%Y")

ALL_TOOLS = {
    "search_documents": FunctionTool.from_defaults(fn=search_documents),
    "list_documents":   FunctionTool.from_defaults(fn=list_documents),
    "calculate":        FunctionTool.from_defaults(fn=calculate),
    "get_current_date": FunctionTool.from_defaults(fn=get_current_date),
}

# ── Skills ────────────────────────────────────────────────────────────────────
SKILLS_DIR = pathlib.Path(__file__).parent / "skills"

def _load_skills() -> dict[str, dict]:
    skills: dict[str, dict] = {}
    if SKILLS_DIR.exists():
        for f in SKILLS_DIR.glob("*.json"):
            data = json.loads(f.read_text(encoding="utf-8"))
            skills[data["name"]] = data
    return skills

SKILLS = _load_skills()

def route_skill(question: str) -> str:
    """Keyword-based router: picks the most specific skill or falls back to general."""
    q = question.lower()
    for name, data in SKILLS.items():
        if name == "general":
            continue
        if any(kw in q for kw in data.get("keywords", [])):
            return name
    return "general"

# ── Agent builder ─────────────────────────────────────────────────────────────
# One AgentWorkflow per skill, built lazily and reused across sessions
_skill_agents: dict[str, AgentWorkflow] = {}

def get_skill_agent(skill_name: str) -> AgentWorkflow:
    if skill_name not in _skill_agents:
        skill      = SKILLS.get(skill_name, SKILLS.get("general", {}))
        tool_names = skill.get("tools", list(ALL_TOOLS.keys()))
        tools      = [ALL_TOOLS[t] for t in tool_names if t in ALL_TOOLS]
        system_prompt = skill.get("system_prompt", "Sei un assistente utile.")
        _skill_agents[skill_name] = AgentWorkflow.from_tools_or_functions(
            tools,
            llm=llm,
            system_prompt=system_prompt,
            verbose=True,
        )
    return _skill_agents[skill_name]

# ── Session store (memory only) ────────────────────────────────────────────────
# Memory is per-session; skill is re-routed per question so one session
# can use multiple skills depending on what is asked.
_session_memory: dict[str, ChatMemoryBuffer] = {}

def get_memory(session_id: str) -> ChatMemoryBuffer:
    if session_id not in _session_memory:
        _session_memory[session_id] = ChatMemoryBuffer.from_defaults(
            token_limit=MEMORY_TOKEN_LIMIT
        )
    return _session_memory[session_id]

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/debug")
async def debug(q: str):
    contexts = retrieve_context(q)
    return {
        "num_contexts": len(contexts),
        "contexts": [
            {"score": round(c["score"], 4), "file": c["file"], "text": c["text"][:300]}
            for c in contexts
        ],
    }

@app.get("/skills")
async def list_skills():
    return {
        name: {"description": data.get("description"), "tools": data.get("tools")}
        for name, data in SKILLS.items()
    }

@app.post("/chat")
async def chat(query: Query):
    skill_name = route_skill(query.question)
    agent      = get_skill_agent(skill_name)
    memory     = get_memory(query.session_id)
    try:
        handler = agent.run(user_msg=query.question, memory=memory)
        result  = await handler
        return {"answer": result.response, "session_id": query.session_id, "skill": skill_name}
    except Exception as e:
        traceback.print_exc()
        return {"answer": f"ERRORE: {e}"}

@app.post("/chat/stream")
async def chat_stream(query: Query):
    skill_name = route_skill(query.question)
    agent      = get_skill_agent(skill_name)
    memory     = get_memory(query.session_id)

    async def generate():
        try:
            handler = agent.run(user_msg=query.question, memory=memory)
            async for event in handler.stream_events():
                if isinstance(event, AgentStream) and event.delta:
                    yield json.dumps({"token": event.delta}) + "\n"
        except Exception as e:
            traceback.print_exc()
            yield json.dumps({"error": str(e)}) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")

@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    _session_memory.pop(session_id, None)
    return {"cleared": session_id}

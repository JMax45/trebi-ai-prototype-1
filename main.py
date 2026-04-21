from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from llama_index.llms.ollama import Ollama
from llama_index.core.agent import AgentWorkflow
from llama_index.core.agent.workflow.base_agent import AgentStream, AgentOutput
from llama_index.core.tools import FunctionTool
from llama_index.core.memory import ChatMemoryBuffer
from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, FusionQuery, Fusion, SparseVector
import json
import pathlib
import re as _re
import sympy
import traceback
from datetime import date

# ── Config ────────────────────────────────────────────────────────────────────
COLLECTION         = "documents"
HYBRID_LIMIT       = 40    # candidates per search leg (dense + sparse)
RERANK_TOP_K       = 8     # parent chunks sent to LLM after reranking
MEMORY_TOKEN_LIMIT = 4096  # sliding-window conversation memory per session
OUTPUT_DIR         = pathlib.Path(__file__).parent / "output"
TEMPLATES_DIR      = pathlib.Path(__file__).parent / "templates"
OUTPUT_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR.mkdir(exist_ok=True)
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

# ── Report extraction ────────────────────────────────────────────────────────
REPORT_START_MARKER = "<<<REPORT_START>>>"
REPORT_END_MARKER   = "<<<REPORT_END>>>"

def check_and_save_report(response: str) -> tuple[str, str | None]:
    """If the response contains a <<<REPORT_START>>>/<<<REPORT_END>>> block, save it
    to the output directory and return (display_text, filename). Otherwise return
    (response, None) unchanged."""
    if REPORT_START_MARKER not in response:
        return response, None
    try:
        s = response.index(REPORT_START_MARKER) + len(REPORT_START_MARKER)
        e = response.index(REPORT_END_MARKER, s)
        block = response[s:e].strip()

        # First line may be FILENAME: <name>
        first_line, _, rest = block.partition("\n")
        if first_line.upper().startswith("FILENAME:"):
            raw_name = first_line.split(":", 1)[1].strip()
            safe_name = pathlib.Path(raw_name).name
            if not safe_name or pathlib.Path(safe_name).suffix.lower() not in {".txt", ".md"}:
                safe_name = f"report_{date.today().strftime('%Y%m%d')}.txt"
            content = rest.strip()
        else:
            safe_name = f"report_{date.today().strftime('%Y%m%d')}.txt"
            content = block

        dest = OUTPUT_DIR / safe_name
        dest.write_text(content, encoding="utf-8")
        display = content + f"\n\n[File salvato: output/{safe_name} — {len(content)} caratteri]"
        return display, safe_name
    except (ValueError, IndexError, OSError) as exc:
        traceback.print_exc()
        return response, None

def read_template(template_name: str) -> str:
    """Legge un template dalla cartella templates/. Passa solo il nome del file (es. report.txt). Il template può contenere segnaposto come {{titolo}}, {{data}}, {{contenuto}} da riempire."""
    safe_name = pathlib.Path(template_name).name
    # Auto-append .txt if no extension provided
    if not pathlib.Path(safe_name).suffix:
        safe_name += ".txt"
    template_path = TEMPLATES_DIR / safe_name
    if not template_path.exists():
        available = [f.name for f in TEMPLATES_DIR.glob("*.txt")] + \
                    [f.name for f in TEMPLATES_DIR.glob("*.md")]
        hint = f"Disponibili: {', '.join(available)}" if available else "Nessun template disponibile."
        return f"Template '{safe_name}' non trovato. {hint}"
    try:
        content = template_path.read_text(encoding="utf-8")
        placeholders = list(dict.fromkeys(_re.findall(r"\{\{(\w+)\}\}", content)))
        placeholder_list = "\n".join(f"- {p}" for p in placeholders)
        return f"{content}\n\n---\nSEGNAPOSTO DA COMPILARE (tutti obbligatori):\n{placeholder_list}"
    except Exception as e:
        return f"Errore nella lettura del template: {e}"

def read_output_file(filename: str) -> str:
    """Legge un file esistente dalla cartella output/. Usa questo strumento per rileggere un file già salvato prima di modificarlo o aggiornarlo. Passa solo il nome del file (es. report.txt)."""
    safe_name = pathlib.Path(filename).name
    if not safe_name or safe_name.startswith("."):
        return "Nome file non valido."
    # Auto-append .txt if no extension provided
    if not pathlib.Path(safe_name).suffix:
        safe_name += ".txt"
    src = OUTPUT_DIR / safe_name
    if not src.exists():
        available = [f.name for f in OUTPUT_DIR.iterdir() if f.is_file()]
        hint = f"File disponibili: {', '.join(available)}" if available else "Nessun file disponibile."
        return f"File '{safe_name}' non trovato. {hint}"
    try:
        return src.read_text(encoding="utf-8")
    except Exception as e:
        return f"Errore nella lettura del file: {e}"

ALL_TOOLS = {
    "search_documents":  FunctionTool.from_defaults(fn=search_documents),
    "list_documents":    FunctionTool.from_defaults(fn=list_documents),
    "calculate":         FunctionTool.from_defaults(fn=calculate),
    "get_current_date":  FunctionTool.from_defaults(fn=get_current_date),
    "read_template":     FunctionTool.from_defaults(fn=read_template),
    "read_output_file":  FunctionTool.from_defaults(fn=read_output_file),
}

# ── Skills ────────────────────────────────────────────────────────────────────
SKILLS_DIR = pathlib.Path(__file__).parent / "skills"

def _load_skills() -> dict[str, dict]:
    skills: dict[str, dict] = {}
    if SKILLS_DIR.exists():
        for f in SKILLS_DIR.glob("*.json"):
            data = json.loads(f.read_text(encoding="utf-8-sig"))
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

# ── Session store (memory + skill lock) ───────────────────────────────────────
_session_memory: dict[str, ChatMemoryBuffer] = {}
_session_skill:  dict[str, str] = {}

def get_memory(session_id: str) -> ChatMemoryBuffer:
    if session_id not in _session_memory:
        _session_memory[session_id] = ChatMemoryBuffer.from_defaults(
            token_limit=MEMORY_TOKEN_LIMIT
        )
    return _session_memory[session_id]

def get_session_skill(session_id: str, question: str) -> str:
    """Route skill for this question; lock to a non-general skill once matched."""
    routed = route_skill(question)
    if routed != "general":
        _session_skill[session_id] = routed
        return routed
    return _session_skill.get(session_id, "general")

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

@app.get("/files")
async def list_output_files():
    files = sorted(OUTPUT_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    return [
        {"name": f.name, "size_bytes": f.stat().st_size,
         "modified": f.stat().st_mtime}
        for f in files if f.is_file()
    ]

@app.post("/chat")
async def chat(query: Query):
    skill_name = get_session_skill(query.session_id, query.question)
    agent      = get_skill_agent(skill_name)
    memory     = get_memory(query.session_id)
    try:
        handler = agent.run(user_msg=query.question, memory=memory)
        result  = await handler
        display, saved_file = check_and_save_report(result.response)
        resp = {"answer": display, "session_id": query.session_id, "skill": skill_name}
        if saved_file:
            resp["saved_file"] = f"output/{saved_file}"
        return resp
    except Exception as e:
        traceback.print_exc()
        return {"answer": f"ERRORE: {e}"}

@app.post("/chat/stream")
async def chat_stream(query: Query):
    skill_name = get_session_skill(query.session_id, query.question)
    agent      = get_skill_agent(skill_name)
    memory     = get_memory(query.session_id)

    async def generate():
        try:
            handler = agent.run(user_msg=query.question, memory=memory)
            step_buffer: list[str] = []
            final_response = ""

            async for event in handler.stream_events():
                if isinstance(event, AgentStream) and event.delta:
                    step_buffer.append(event.delta)
                elif isinstance(event, AgentOutput):
                    if event.tool_calls:
                        for tc in event.tool_calls:
                            yield json.dumps({"tool_call": tc.tool_name, "args": tc.tool_kwargs}) + "\n"
                    else:
                        # Overwrite each time — only the last one is the real final answer
                        final_response = "".join(step_buffer)
                    step_buffer = []

            display_response, saved_file = check_and_save_report(final_response)
            for token in display_response:
                yield json.dumps({"token": token}) + "\n"
            if saved_file:
                yield json.dumps({"saved": f"output/{saved_file}"}) + "\n"

        except Exception as e:
            traceback.print_exc()
            yield json.dumps({"error": str(e)}) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")

@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    _session_memory.pop(session_id, None)
    return {"cleared": session_id}

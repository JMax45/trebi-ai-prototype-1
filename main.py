from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from llama_index.llms.ollama import Ollama
from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, FusionQuery, Fusion, SparseVector
import json
import traceback

# ── Config ────────────────────────────────────────────────────────────────────
COLLECTION   = "documents"
HYBRID_LIMIT = 40   # candidates retrieved per search leg (dense + sparse)
RERANK_TOP_K = 8    # parent chunks kept after reranking → sent to LLM

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

    # Hybrid search: dense (semantic) + sparse (BM25-like) fused with RRF
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

    # Deduplicate by parent_id (first = highest RRF rank)
    seen: dict[str, object] = {}
    for hit in hits:
        pid = hit.payload["parent_id"]
        if pid not in seen:
            seen[pid] = hit
    unique = list(seen.values())

    if not unique:
        return []

    # Cross-encoder rerank: scores each (query, parent_text) pair precisely
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

# ── Prompt ────────────────────────────────────────────────────────────────────
def build_prompt(question: str, contexts: list[dict]) -> str:
    parts = [
        f"[Documento {i+1} — {c['file']}]\n{c['text']}"
        for i, c in enumerate(contexts)
    ]
    return (
        "Sei un assistente che risponde alle domande basandosi SOLO sui documenti forniti.\n"
        "REGOLE:\n"
        "1. Usa solo le informazioni presenti nei documenti. Non inventare fatti.\n"
        "2. Se l'informazione richiesta è genuinamente assente, rispondi: "
        "'Informazione non trovata nei documenti.'\n"
        "3. Rispondi con frasi complete e chiare.\n"
        "4. Non aggiungere frasi di chiusura.\n\n"
        + "\n\n".join(parts)
        + f"\n\nDomanda: {question}\nRisposta:"
    )

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

@app.post("/chat")
async def chat(query: Query):
    contexts = retrieve_context(query.question)
    if not contexts:
        return {"answer": "Nessun documento rilevante trovato."}
    try:
        response = llm.complete(build_prompt(query.question, contexts))
        return {"answer": response.text, "sources": list({c["file"] for c in contexts})}
    except Exception as e:
        traceback.print_exc()
        return {"answer": f"ERRORE LLM: {e}"}

@app.post("/chat/stream")
async def chat_stream(query: Query):
    contexts = retrieve_context(query.question)
    if not contexts:
        async def empty():
            yield json.dumps({"answer": "Nessun documento rilevante trovato."}) + "\n"
        return StreamingResponse(empty(), media_type="application/x-ndjson")

    sources = list({c["file"] for c in contexts})
    prompt  = build_prompt(query.question, contexts)

    async def generate():
        yield json.dumps({"source": ", ".join(sources)}) + "\n"
        try:
            for chunk in llm.stream_complete(prompt):
                if chunk.delta:
                    yield json.dumps({"token": chunk.delta}) + "\n"
        except Exception as e:
            traceback.print_exc()
            yield json.dumps({"error": str(e)}) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")

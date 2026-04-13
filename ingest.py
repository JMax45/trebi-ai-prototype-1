import pathlib
import uuid

from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, SparseVectorParams,
    SparseIndexParams, PointStruct, SparseVector,
)
from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document
from llama_index.readers.file import PDFReader
from bs4 import BeautifulSoup

# ── Config ────────────────────────────────────────────────────────────────────
COLLECTION     = "documents"
DENSE_DIM      = 1024
PARENT_SIZE    = 1500   # chars — returned to LLM
PARENT_OVERLAP = 150
CHILD_SIZE     = 400    # chars — used for retrieval
CHILD_OVERLAP  = 40
BATCH_SIZE     = 16

# ── Embedding model ───────────────────────────────────────────────────────────
print("Loading BGE-M3 …")
embed_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

# ── Qdrant ────────────────────────────────────────────────────────────────────
qdrant = QdrantClient("localhost", port=6333)

if qdrant.collection_exists(COLLECTION):
    qdrant.delete_collection(COLLECTION)

qdrant.create_collection(
    collection_name=COLLECTION,
    vectors_config={"dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE)},
    sparse_vectors_config={
        "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))
    },
)

# ── Document loading ──────────────────────────────────────────────────────────
def load_documents() -> list[Document]:
    docs: list[Document] = []

    # PDFs — SimpleDirectoryReader sets file_name metadata automatically
    pdf_docs = SimpleDirectoryReader(
        "./docs",
        required_exts=[".pdf"],
        file_extractor={".pdf": PDFReader()},
    ).load_data()
    docs.extend(pdf_docs)
    print(f"  PDF pages: {len(pdf_docs)}")

    for p in pathlib.Path("./docs").glob("*.html"):
        soup = BeautifulSoup(p.read_text(encoding="utf-8", errors="ignore"), "html.parser")
        docs.append(Document(text=soup.get_text(separator="\n", strip=True),
                             metadata={"file_name": p.name}))
        print(f"  HTML: {p.name}")

    for p in pathlib.Path("./docs").glob("*.txt"):
        docs.append(Document(text=p.read_text(encoding="utf-8", errors="ignore"),
                             metadata={"file_name": p.name}))
        print(f"  TXT: {p.name}")

    return docs

# ── Chunking ──────────────────────────────────────────────────────────────────
def split_text(text: str, size: int, overlap: int) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks

def build_child_chunks(docs: list[Document]) -> list[dict]:
    """Parent-child chunking: small child chunks for retrieval,
    parent text stored in payload so the LLM gets richer context."""
    children = []
    for doc in docs:
        text = doc.text.strip()
        if not text:
            continue
        file_name = doc.metadata.get("file_name", "unknown")
        for parent_text in split_text(text, PARENT_SIZE, PARENT_OVERLAP):
            parent_text = parent_text.strip()
            if not parent_text:
                continue
            parent_id = str(uuid.uuid4())
            for child_text in split_text(parent_text, CHILD_SIZE, CHILD_OVERLAP):
                child_text = child_text.strip()
                if len(child_text) < 20:
                    continue
                children.append({
                    "id":          str(uuid.uuid4()),
                    "child_text":  child_text,
                    "parent_text": parent_text,
                    "parent_id":   parent_id,
                    "file_name":   file_name,
                })
    return children

# ── Embed & index ─────────────────────────────────────────────────────────────
def embed_and_index(chunks: list[dict]) -> None:
    total = len(chunks)
    print(f"Embedding {total} child chunks …")
    points = []

    for i in range(0, total, BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        output = embed_model.encode(
            [c["child_text"] for c in batch],
            batch_size=BATCH_SIZE,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        for j, chunk in enumerate(batch):
            sw = output["lexical_weights"][j]   # {token_id: weight}
            points.append(PointStruct(
                id=chunk["id"],
                vector={
                    "dense":  output["dense_vecs"][j].tolist(),
                    "sparse": SparseVector(
                        indices=[int(k) for k in sw],
                        values=[float(v) for v in sw.values()],
                    ),
                },
                payload={
                    "child_text":  chunk["child_text"],
                    "parent_text": chunk["parent_text"],
                    "parent_id":   chunk["parent_id"],
                    "file_name":   chunk["file_name"],
                },
            ))
        print(f"  {min(i + BATCH_SIZE, total)}/{total}", end="\r")

    print(f"\nUploading {len(points)} points …")
    UPLOAD_BATCH = 100
    for i in range(0, len(points), UPLOAD_BATCH):
        qdrant.upsert(COLLECTION, points[i:i + UPLOAD_BATCH])

# ── Run ───────────────────────────────────────────────────────────────────────
print("Loading documents …")
docs = load_documents()
print(f"Total sections loaded: {len(docs)}")

chunks = build_child_chunks(docs)
print(f"Total child chunks: {len(chunks)}")

embed_and_index(chunks)
print(f"\nDone — {qdrant.count(COLLECTION).count} points in '{COLLECTION}'.")

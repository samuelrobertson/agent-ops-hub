# app/api/main.py
# FastAPI app: exposes /ingest and /ask. Wires VectorStore + LangGraph.

import os
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel

from app.core.vector.store import VectorStore
from app.core.chains.embedding import fetch_text_from_url, chunk_text, embed_chunks
from app.core.graph.qa_graph import build_graph
from app.core.skills.calendar_skill import get_busy_slots


app = FastAPI(title="Agent Ops Hub")

# Storage configuration (can be overridden via Deployment Secrets)
DATA_DIR = os.getenv("DATA_DIR", "data")
INDEX_PATH = os.path.join(DATA_DIR, "index.faiss")
DOCSTORE_PATH = os.path.join(DATA_DIR, "docstore.jsonl")

# Ensure the data directory exists (VectorStore also guards this, but cheap to do here)
os.makedirs(DATA_DIR, exist_ok=True)

# Single vector store instance shared by endpoints
# NOTE: VectorStore requires (data_dir, index_path, docstore_path)
vs = VectorStore(DATA_DIR, INDEX_PATH, DOCSTORE_PATH)

# Build the LangGraph once at startup (it captures the vector store reference)
graph = build_graph(vs)


class IngestRequest(BaseModel):
    url: Optional[str] = None


class AskRequest(BaseModel):
    question: str


@app.post("/ingest")
async def ingest(req: IngestRequest):
    """
    Ingest from a URL:
      1) Fetch raw text (robust fetcher)
      2) Chunk → Embed
      3) Upsert into FAISS + JSONL
    Returns how many chunks were indexed and total corpus size.
    """
    if not req.url:
        return {"indexed": 0, "docs_total": vs.size()}

    # 1) Fetch raw text from the URL
    text = await fetch_text_from_url(req.url)

    # Guard: if a site returns no text (blocked/empty), bail out gracefully
    if not text or not text.strip():
        return {"indexed": 0, "docs_total": vs.size()}

    # 2) Chunk and embed
    chunks = chunk_text(text)
    if not chunks:
        return {"indexed": 0, "docs_total": vs.size()}
    embeddings = await embed_chunks(chunks)

    # 3) Upsert vectors + metadata
    meta = {"url": req.url, "title": req.url}
    ids = vs.upsert(chunks, embeddings, meta)

    return {"indexed": len(ids), "docs_total": vs.size()}


@app.post("/ask")
async def ask(req: AskRequest):
    """
    Ask a question:
      - If it looks like a scheduling request, route to our stub calendar skill
      - Otherwise run the LangGraph QA path: plan → retrieve → answer
    Returns answer + citations + route.
    """
    q = (req.question or "").strip()

    # Simple route to prove skills path works
    if q.lower().startswith("schedule"):
        return {
            "answer": get_busy_slots(),
            "skill": "calendar",
            "route": "skill",
        }

    # Run the compiled LangGraph on the QA path
    result = await graph.ainvoke({"question": q})
    # result should include 'answer', 'citations', 'route'
    return result


@app.get("/healthz")
async def health():
    """
    Lightweight health check so you (and Replit) can verify the API is up.
    """
    return {
        "ok": True,
        "vectors": vs.size(),
        "data_dir": DATA_DIR,
        "index_path": INDEX_PATH,
        "docstore_path": DOCSTORE_PATH,
    }
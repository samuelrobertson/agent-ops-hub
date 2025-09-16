from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from app.core.vector.store import VectorStore
from app.core.chains.embedding import embed_chunks, chunk_text, fetch_text_from_url
from app.core.graph.qa_graph import build_graph
from app.core.skills.calendar_skill import get_busy_slots
import os

app = FastAPI()

class IngestRequest(BaseModel):
    url: str | None = None

class AskRequest(BaseModel):
    question: str

vs = VectorStore(
    data_dir=os.getenv("DATA_DIR", "./data"),
    index_path=os.getenv("VECTOR_INDEX", "./data/index.faiss"),
    docstore_path=os.getenv("DOCSTORE", "./data/docstore.jsonl")
)

graph = build_graph(vs)

@app.post("/ingest")
async def ingest(req: IngestRequest = None, file: UploadFile = File(None)):
    texts = []
    if req and req.url:
        txt = await fetch_text_from_url(req.url)
        texts.append({"text": txt, "meta": {"url": req.url, "title": req.url}})
    if file:
        b = await file.read()
        texts.append({"text": b.decode("utf-8", errors="ignore"), "meta": {"filename": file.filename}})
    count = 0
    for item in texts:
        chunks = chunk_text(item["text"])
        embs = await embed_chunks([c["text"] for c in chunks])
        ids = vs.upsert(chunks, embs, item["meta"])
        count += len(ids)
    return {"indexed": count, "docs_total": vs.size()}

@app.post("/ask")
async def ask(req: AskRequest):
    q = req.question.strip()
    if q.lower().startswith("schedule"):
        slots = get_busy_slots()
        return {"answer": f"You are busy {slots}.", "skill": "calendar", "route": "skill"}
    result = await graph.ainvoke({"question": q})
    return {"answer": result["answer"], "citations": result.get("citations", []), "route": "qa"}

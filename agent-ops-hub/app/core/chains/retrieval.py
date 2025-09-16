from typing import List, Dict
from app.core.vector.store import VectorStore
from app.core.chains.embedding import embed_query

async def retrieve(vs: VectorStore, question: str, k: int = 5) -> List[Dict]:
    qemb = await embed_query(question)
    return vs.search(qemb, k=k)

# app/core/chains/embedding.py
# Utilities for fetching text from URLs, chunking text, and creating embeddings.

import os
import json
from typing import List
import numpy as np
import httpx
from langchain.text_splitter import RecursiveCharacterTextSplitter


def chunk_text(text: str, chunk_size: int = 1200, chunk_overlap: int = 200) -> List[str]:
    """
    Split long text into overlapping chunks so retrieval preserves context.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(text or "")


async def fetch_text_from_url(url: str) -> str:
    """
    Robust URL fetcher:
      - Sets a User-Agent (some sites block default clients)
      - Follows redirects
      - Handles JSON sources (tries common fields like 'content', 'text', 'extract', 'body', 'summary', 'description')
      - Falls back to r.text for HTML/markdown/plaintext
      - As a last resort, decodes bytes as UTF-8 (ignoring errors)
    Returns the raw textual content to be chunked/embedded.
    """
    async with httpx.AsyncClient(
        timeout=60,
        headers={"User-Agent": "AgentOpsHub/1.0 (+replit)"},
        follow_redirects=True,
    ) as client:
        r = await client.get(url)
        ct = (r.headers.get("content-type") or "").lower()

        if "application/json" in ct:
            try:
                data = r.json()
            except Exception:
                return r.text
            if isinstance(data, dict):
                for k in ("content", "text", "extract", "body", "summary", "description"):
                    v = data.get(k)
                    if isinstance(v, str) and v.strip():
                        return v
            return json.dumps(data, ensure_ascii=False)

        if any(t in ct for t in ("text/", "html", "markdown")):
            return r.text

        try:
            return r.content.decode("utf-8", errors="ignore")
        except Exception:
            return ""


async def _embed_batch(texts: List[str]) -> List[List[float]]:
    """
    Call the OpenAI embeddings endpoint once for the whole batch of texts.
    """
    base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    url = f"{base}/embeddings"
    model = os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small")
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(url, headers=headers, json={"model": model, "input": texts})
        
        if not resp.is_success:
            # Log the specific error details
            try:
                error_data = resp.json()
                print(f"OpenAI API Error: {resp.status_code} - {error_data}")
            except:
                print(f"OpenAI API Error: {resp.status_code} - {resp.text}")
            resp.raise_for_status()
            
        data = resp.json()
        return [item["embedding"] for item in data["data"]]


async def embed_chunks(texts: List[str]) -> np.ndarray:
    """
    Embed a list of chunk strings → NumPy array [n_chunks, dim].
    Returns an empty (0, dim) array if there are no texts.
    Processes in batches to avoid OpenAI token limits.
    """
    if not texts:
        return np.zeros((0, 1536), dtype=np.float32)

    # Process in smaller batches to avoid token limits
    # Assuming ~4 chars per token, and 1200 char chunks, that's ~300 tokens per chunk
    # With 300k token limit, we can safely do ~800 chunks per batch
    # Let's be conservative and use 100 chunks per batch
    batch_size = 100
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        print(f"Processing embedding batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size} with {len(batch)} chunks")
        
        batch_vecs = await _embed_batch(batch)
        all_embeddings.extend(batch_vecs)
    
    arr = np.asarray(all_embeddings, dtype=np.float32)
    return arr


async def embed_query(text: str) -> np.ndarray:
    """
    Convenience: embed a single query string → (dim,) vector.
    """
    arr = await embed_chunks([text])
    return arr[0]
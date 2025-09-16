import os
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
import httpx
import tiktoken
import json

def chunk_text(text: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    chunks = splitter.split_text(text)
    return [{"text": c} for c in chunks]

async def embed_chunks(texts: List[str]) -> List[List[float]]:
    key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-large")
    url = "https://api.openai.com/v1/embeddings"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    out = []
    async with httpx.AsyncClient(timeout=60) as client:
        for t in texts:
            payload = {"model": model, "input": t}
            r = await client.post(url, headers=headers, json=payload)
            data = r.json()
            out.append(data["data"][0]["embedding"])
    return out

async def embed_query(text: str) -> List[float]:
    return (await embed_chunks([text]))[0]

async def fetch_text_from_url(url: str) -> str:
    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        r = await client.get(url)
        return r.text

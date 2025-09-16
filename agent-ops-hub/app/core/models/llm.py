import os, httpx

async def chat_answer(prompt: str) -> str:
    key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("CHAT_MODEL", "gpt-4o-mini")
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages":[{"role":"system","content":"Answer using only the provided context. If the answer is not in the context, say you lack sufficient information and avoid guessing."},{"role":"user","content":prompt}]}
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(url, headers=headers, json=payload)
        data = r.json()
        return data["choices"][0]["message"]["content"]

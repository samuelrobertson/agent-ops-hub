import argparse, asyncio, httpx

def main():
    p = argparse.ArgumentParser()
    p.add_argument("question")
    p.add_argument("--host", default="http://localhost:8000")
    args = p.parse_args()
    asyncio.run(run(args.host, args.question))

async def run(host, q):
    async with httpx.AsyncClient() as client:
        r = await client.post(host + "/ask", json={"question": q})
        print(r.text)

if __name__ == "__main__":
    main()

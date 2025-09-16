import argparse, asyncio, json, sys, httpx

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url")
    p.add_argument("--file")
    p.add_argument("--host", default="http://localhost:8000")
    args = p.parse_args()
    if args.url:
        asyncio.run(post_json(args.host + "/ingest", {"url": args.url}))
    elif args.file:
        asyncio.run(post_file(args.host + "/ingest", args.file))
    else:
        print("provide --url or --file")
        sys.exit(1)

async def post_json(url, data):
    async with httpx.AsyncClient() as client:
        r = await client.post(url, json=data)
        print(r.text)

async def post_file(url, path):
    async with httpx.AsyncClient() as client:
        with open(path, "rb") as f:
            files = {"file": (path, f, "application/octet-stream")}
            r = await client.post(url, files=files)
            print(r.text)

if __name__ == "__main__":
    main()

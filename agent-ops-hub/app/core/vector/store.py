import os, json
import numpy as np
import faiss

class VectorStore:
    def __init__(self, data_dir: str, index_path: str, docstore_path: str):
        self.data_dir = data_dir
        self.index_path = index_path
        self.docstore_path = docstore_path
        os.makedirs(self.data_dir, exist_ok=True)
        self.dim = None
        self.index = None
        self.docmeta = []
        self._load()

    def _load(self):
        if os.path.exists(self.docstore_path):
            with open(self.docstore_path, "r", encoding="utf-8") as f:
                self.docmeta = [json.loads(l) for l in f]
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            if self.docmeta:
                vec = self._vector_dim_from_index()
                self.dim = vec
        else:
            self.index = None

    def _vector_dim_from_index(self):
        return self.index.d

    def upsert(self, chunks, embeddings, parent_meta):
        arr = np.array(embeddings).astype("float32")
        if self.index is None:
            self.dim = arr.shape[1]
            self.index = faiss.IndexFlatIP(self.dim)
        ids = []
        for i, emb in enumerate(arr):
            meta = {"id": f"doc-{len(self.docmeta)+1}", "text": chunks[i]["text"]}
            meta.update(parent_meta)
            self.docmeta.append(meta)
            ids.append(meta["id"])
        faiss.normalize_L2(arr)
        self.index.add(arr)
        faiss.write_index(self.index, self.index_path)
        with open(self.docstore_path, "w", encoding="utf-8") as f:
            for m in self.docmeta:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        return ids

    def search(self, query_emb, k=5):
        if self.index is None:
            return []
        q = np.array([query_emb]).astype("float32")
        faiss.normalize_L2(q)
        D, I = self.index.search(q, k)
        results = []
        for rank, idx in enumerate(I[0]):
            if idx < 0 or idx >= len(self.docmeta):
                continue
            m = self.docmeta[idx]
            results.append({"id": m["id"], "title": m.get("title") or m.get("filename") or m["id"], "url": m.get("url"), "text": m["text"], "score": float(D[0][rank])})
        return results

    def size(self):
        return len(self.docmeta)

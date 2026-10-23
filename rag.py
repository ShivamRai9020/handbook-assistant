import ollama, json, math, sys, pathlib

EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "llama3.2"
docs = json.loads(pathlib.Path("index.json").read_text())

def embed(text):
    return ollama.embed(model=EMBED_MODEL, input=text)["embeddings"][0]

def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x*x for x in a)) * math.sqrt(sum(y*y for y in b)))

def retrieve(question, k=2):
    q = embed("search_query: " + question)
    return sorted(docs, key=lambda d: cosine(q, d["vector"]), reverse=True)[:k]

def answer(question):
    hits = retrieve(question)
    context = "\n\n".join(f"[{d['file']}]\n{d['text']}" for d in hits)
    resp = ollama.chat(
        model=CHAT_MODEL,
        options={"temperature": 0},
        messages=[
            {"role": "system", "content":
             "Answer ONLY from the job posts in the context. "
             "Cite the file name like [3.txt] for every claim. "
             "If the answer isn't in the context, say \"I don't know.\""},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    print("Retrieved:", [d["file"] for d in hits])
    print(resp["message"]["content"])

if __name__ == "__main__":
    answer(" ".join(sys.argv[1:]))

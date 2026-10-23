import ollama, math, sys, pathlib

EMBED_MODEL, CHAT_MODEL = "nomic-embed-text", "qwen2.5:7b"

def embed(text):
    return ollama.embed(model=EMBED_MODEL, input=text)["embeddings"][0]

def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x*x for x in a)) * math.sqrt(sum(y*y for y in b)))

def chunk(text, size=1000, overlap=40):
    """Split on blank lines (sections); fall back to fixed-size windows for long sections."""
    chunks = []
    for section in [s.strip() for s in text.split("\n\n") if s.strip()]:
        if len(section) <= size:
            chunks.append(section)
        else:
            for start in range(0, len(section), size - overlap):
                chunks.append(section[start:start + size])
    return chunks

text = pathlib.Path("docs/handbook.txt").read_text()
chunks = [{"id": i, "text": c, "vector": embed("search_document: " + c)}
          for i, c in enumerate(chunk(text))]
print(f"{len(chunks)} chunks\n")

question = " ".join(sys.argv[1:])
q = embed("search_query: " + question)
hits = sorted(chunks, key=lambda c: cosine(q, c["vector"]), reverse=True)[:2]
for h in hits:
    print(f"{cosine(q, h['vector']):.3f}  chunk {h['id']}: {h['text'][:60]}...")

context = "\n\n".join(f"[chunk {h['id']}]\n{h['text']}" for h in hits)
resp = ollama.chat(model=CHAT_MODEL, options={"temperature": 0}, messages=[
    {"role": "system", "content": "Answer ONLY from the context. Cite like [chunk 2]. "
                                  "If it isn't there, say \"I don't know.\""},
    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}])
print("\n" + resp["message"]["content"])

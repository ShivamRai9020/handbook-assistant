import ollama, math, json, sys, pathlib

MODEL = "nomic-embed-text"
INDEX = pathlib.Path("index.json")

def embed(text):
    return ollama.embed(model=MODEL, input=text)["embeddings"][0]

def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x*x for x in a)) * math.sqrt(sum(y*y for y in b)))

def build_index(folder="samples"):
    docs = []
    for f in sorted(pathlib.Path(folder).glob("*.txt")):
        text = f.read_text().strip()
        docs.append({"file": f.name, "text": text,
                     "vector": embed("search_document: " + text)})
    INDEX.write_text(json.dumps(docs))
    print(f"Indexed {len(docs)} docs into {INDEX}")

def search(query, k=3):
    docs = json.loads(INDEX.read_text())
    q = embed("search_query: " + query)
    ranked = sorted(docs, key=lambda d: cosine(q, d["vector"]), reverse=True)
    for d in ranked[:k]:
        print(f"{cosine(q, d['vector']):.3f}  {d['file']}  {d['text'][:70]}...")

if __name__ == "__main__":
    if sys.argv[1] == "index":
        build_index()
    else:
        search(" ".join(sys.argv[1:]))

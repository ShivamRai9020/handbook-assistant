import ollama, math

def embed(text):
    return ollama.embed(model="nomic-embed-text", input=text)["embeddings"][0]

def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x*x for x in a)) * math.sqrt(sum(y*y for y in b)))

query = "Backend engineer with Go and Kafka"
others = [
    "Server-side developer, Golang, event streaming",
    "Frontend developer with React",
    "Pastry chef needed",
]

print("WITHOUT prefixes")
q = embed(query)
for t in others:
    print(f"{cosine(q, embed(t)):.3f}  {t}")

print("\nWITH prefixes")
q = embed("search_query: " + query)
for t in others:
    print(f"{cosine(q, embed('search_document: ' + t)):.3f}  {t}")

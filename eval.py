import ollama, math, re, sys, pathlib

EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:7b"

# question, expected chunk id (None = not in docs), keywords that must appear
TESTS = [
    ("How many paid leaves do I get?",            0, ["24"]),
    ("Can I carry forward unused leave?",         0, ["10"]),
    ("When do I need a doctor's note?",           0, ["2 days"]),
    ("How many days can I work from home?",       1, ["3"]),
    ("Can I work fully remote?",                  1, ["HR"]),
    ("What laptop do new joiners get?",           2, ["MacBook"]),
    ("What if I lose my laptop?",                 2, ["24 hours"]),
    ("When is salary credited?",                  3, ["last working day"]),
    ("Who approves international travel?",        4, ["VP"]),
    ("What is the dress code?",                None, ["don't know"]),
    ("Is sick leave paid, and when do I need proof?",      0, ["2 days"]),
    ("Do I need approval to fly to Delhi?",                4, ["portal"]),
    ("If I lose my MacBook, who do I tell?",               2, ["IT"]),
    ("When will my travel expenses be paid back?",         3, ["next salary"]),
    ("What is the company's maternity leave policy?",   None, ["don't know"]),
]

def embed(t):
    return ollama.embed(model=EMBED_MODEL, input=t)["embeddings"][0]

def cosine(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x*x for x in a)) * math.sqrt(sum(y*y for y in b)))

chunks = [s.strip() for s in pathlib.Path("docs/handbook.txt").read_text().split("\n\n") if s.strip()]
vecs = [embed("search_document: " + c) for c in chunks]

def rag(question, k=2):
    q = embed("search_query: " + question)
    hits = sorted(range(len(chunks)), key=lambda i: cosine(q, vecs[i]), reverse=True)[:k]
    context = "\n\n".join(f"<chunk id={i}>\n{chunks[i]}\n</chunk id={i}>" for i in hits)
    resp = ollama.chat(model=CHAT_MODEL, options={"temperature": 0}, messages=[
        {"role": "system", "content": "Use the context to answer. After each fact, cite the chunk that contains that exact fact, like [chunk N]. "
"Say \"I don't know\" only if the context has nothing relevant."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}])
    return hits, resp["message"]["content"]

score = {"retrieval": 0, "answer": 0, "citation": 0}
in_docs = sum(1 for _, c, _ in TESTS if c is not None)

for question, expected, keywords in TESTS:
    hits, answer = rag(question)
    cited = {int(n) for n in re.findall(r"chunk (?:id=)?\[?(\d+)\]?", answer)}
    ans_ok = all(k.lower() in answer.lower().replace("’", "'") for k in keywords)
    if expected is None:
        ret_ok = cite_ok = True   # nothing to retrieve or cite
    else:
        ret_ok = expected in hits
        cite_ok = expected in cited
        score["retrieval"] += ret_ok
        score["citation"] += cite_ok
    score["answer"] += ans_ok
    marks = "".join("✅" if ok else "❌" for ok in (ret_ok, ans_ok, cite_ok))
    print(f"{marks}  {question}")
    if not (ans_ok and cite_ok):
        print(f"      -> {answer.strip()[:120]}")

print(f"\nModel: {CHAT_MODEL}")
print(f"Retrieval: {score['retrieval']}/{in_docs}  Answer: {score['answer']}/{len(TESTS)}  Citation: {score['citation']}/{in_docs}")

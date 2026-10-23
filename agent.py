import ollama, math, sys, pathlib, json
from extract import extract

CHAT_MODEL, EMBED_MODEL = "qwen2.5:7b", "nomic-embed-text"

# ---------- tools ----------
def calculator(expression: str) -> str:
    """Safely evaluate a math expression like '24 * 12 / 5'."""
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return "error: only numbers and + - * / ( ) allowed"
    return str(eval(expression))

def embed(t):
    return ollama.embed(model=EMBED_MODEL, input=t)["embeddings"][0]

def cosine(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x*x for x in a)) * math.sqrt(sum(y*y for y in b)))

HANDBOOK = [s.strip() for s in pathlib.Path("docs/handbook.txt").read_text().split("\n\n") if s.strip()]
HB_VECS = [embed("search_document: " + s) for s in HANDBOOK]

def search_handbook(query: str) -> str:
    """Search the company handbook for policies on leave, WFH, laptops, payroll, travel."""
    q = embed("search_query: " + query)
    best = sorted(range(len(HANDBOOK)), key=lambda i: cosine(q, HB_VECS[i]), reverse=True)[:2]
    return "\n\n".join(HANDBOOK[i] for i in best)

def list_jobs() -> str:
    """List all open job posts with their file names."""
    return "\n".join(f"{f.name}: {f.read_text().strip()}" for f in sorted(pathlib.Path("samples").glob("*.txt")))

def count_jobs_by_seniority() -> str:
    """Count open job posts by seniority level (junior, mid, senior, unknown)."""
    counts = {"junior": [], "mid": [], "senior": [], "unknown": []}
    for f in sorted(pathlib.Path("samples").glob("*.txt")):
        data, _ = extract(f.read_text())
        level = data.get("seniority", "unknown")
        if level not in counts:
            level = "unknown"
        counts[level].append(f"{f.name} ({data.get('title', '?')})")
    return "\n".join(f"{lvl}: {len(jobs)} -> {', '.join(jobs) or 'none'}"
                     for lvl, jobs in counts.items())

TOOLS = {"calculator": calculator, "search_handbook": search_handbook,
         "list_jobs": list_jobs, "count_jobs_by_seniority": count_jobs_by_seniority}

# ---------- agent loop ----------
def run(question, max_steps=5):
    messages = [
        {"role": "system", "content": "You are a helpful HR assistant. Use tools when needed. "
                                      "Use the calculator for any arithmetic."},
        {"role": "user", "content": question},
    ]
    for step in range(max_steps):
        resp = ollama.chat(model=CHAT_MODEL, messages=messages,
                           tools=list(TOOLS.values()), options={"temperature": 0})
        msg = resp["message"]
        messages.append(msg)
        if not msg.get("tool_calls"):
            print("\nAnswer:", msg["content"])
            return
        for call in msg["tool_calls"]:
            name, args = call["function"]["name"], call["function"]["arguments"]
            print(f"[step {step+1}] calling {name}({json.dumps(args)})")
            result = TOOLS[name](**args) if name in TOOLS else f"unknown tool {name}"
            messages.append({"role": "tool", "content": result, "tool_name": name})
    print("Stopped: too many steps.")

if __name__ == "__main__":
    run(" ".join(sys.argv[1:]))

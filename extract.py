import json, sys, pathlib
import ollama

MODEL = "llama3.2"

SYSTEM = """You extract structured data from job descriptions.
Reply with ONLY valid JSON, no extra text, in exactly this shape:
{"title": string, "company": string, "skills": [string], "seniority": "junior" | "mid" | "senior" | "unknown"}
Rules:
- If the company isn't mentioned, use "NA".
- If seniority isn't clear, use "unknown".
- Only include skills explicitly stated; don't infer.
- Example: "Kafka is required [1.txt]."""

def extract(text, temperature=0):
    resp = ollama.chat(
        model=MODEL,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": text}],
        format="json",
        options={"temperature": temperature},
    )
    raw = resp["message"]["content"].strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {"error": "invalid JSON", "raw": raw}, resp
    missing = {"title", "company", "skills", "seniority"} - data.keys()
    if missing:
        data["warning"] = f"missing fields: {sorted(missing)}"
    return data, resp

if __name__ == "__main__":
    folder = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "samples")
    temp = float(sys.argv[2]) if len(sys.argv) > 2 else 0
    for f in sorted(folder.glob("*.txt")):
        data, resp = extract(f.read_text(), temp)
        print(f"\n=== {f.name} ({resp.get('prompt_eval_count')} in / {resp.get('eval_count')} out) ===")
        print(json.dumps(data, indent=2))

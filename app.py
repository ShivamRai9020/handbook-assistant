import ollama, math, re, pathlib
import streamlit as st

def best_source(answer, hits):
    words = set(re.findall(r"\w+", answer.lower()))
    return max(hits, key=lambda i: len(words & set(re.findall(r"\w+", chunks[i].lower()))))

EMBED_MODEL, CHAT_MODEL = "nomic-embed-text", "qwen2.5:7b"

def embed(t):
    return ollama.embed(model=EMBED_MODEL, input=t)["embeddings"][0]

def cosine(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x*x for x in a)) * math.sqrt(sum(y*y for y in b)))

@st.cache_resource          # build the index once, not on every message
def load_index():
    chunks = [s.strip() for s in pathlib.Path("docs/handbook.txt").read_text().split("\n\n") if s.strip()]
    return chunks, [embed("search_document: " + c) for c in chunks]

chunks, vecs = load_index()

st.title("📘 Handbook Assistant")
st.caption(f"Answers from the company handbook · {CHAT_MODEL}")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if question := st.chat_input("Ask about leave, WFH, laptops, payroll, travel..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    q = embed("search_query: " + question)
    hits = sorted(range(len(chunks)), key=lambda i: cosine(q, vecs[i]), reverse=True)[:2]
    context = "\n\n".join(f"<chunk id={i}>\n{chunks[i]}\n</chunk id={i}>" for i in hits)

    with st.chat_message("assistant"):
        stream = ollama.chat(model=CHAT_MODEL, stream=True, options={"temperature": 0}, messages=[
            {"role": "system", "content": "Use the context to answer. After each fact, cite the chunk that contains that exact fact, like [chunk N]. "
"Say \"I don't know\" only if the context has nothing relevant."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}])
        answer = st.write_stream(part["message"]["content"] for part in stream)
        cited = {int(n) for n in re.findall(r"chunk \[?(\d+)\]?", answer)}
        best = best_source(answer, hits)
        if cited and best not in cited:
            st.warning(f"⚠️ The model cited chunk {', '.join(map(str, cited))}, but this answer best matches chunk {best}.")
        with st.expander("Sources"):
            for i in hits:
                st.markdown(f"**chunk {i}**\n\n{chunks[i]}")
    st.session_state.messages.append({"role": "assistant", "content": answer})

import os
import math
import time
from pathlib import Path
from dotenv import load_dotenv
from google import genai

load_dotenv(Path(__file__).parent / ".env")
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-3.8-flash"

# 1. Load notes and split into chunks (one per paragraph)
notes_path = Path(__file__).parent / "notes.txt"
text = notes_path.read_text(encoding="utf-8")
chunks = [c.strip() for c in text.split("\n\n") if c.strip()]

# 2. Embed all chunks in one call
result = client.models.embed_content(model=MODEL, contents=chunks)
chunk_vectors = [e.values for e in result.embeddings]


def similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


# 3. Embed the question and find the best chunks
question = input("Ask a question about my notes: ")

q_result = client.models.embed_content(model=MODEL, contents=question)
q_vec = q_result.embeddings[0].values

scores = []
for i, vec in enumerate(chunk_vectors):
    scores.append((similarity(q_vec, vec), i))

scores.sort(reverse=True)
top = scores[:2]

# 4. Ask Gemini to answer using only those chunks
context = "\n\n".join(chunks[i] for score, i in top)

prompt = f"""Answer the question using ONLY the notes below.
If the answer is not in the notes, say "I don't know based on my notes."

Notes:
{context}

Question: {question}"""

answer = None
for attempt in range(3):
    try:
        reply = client.models.generate_content(
            model=CHAT_MODEL,
            contents=prompt
        )
        answer = reply.text
        break
    except Exception as e:
        print(f"Attempt {attempt + 1} failed: {e}")
        time.sleep(10)

print("\nAnswer:")
print(answer if answer else "No answer. Try again later.")
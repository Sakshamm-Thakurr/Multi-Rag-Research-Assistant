import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

def summariser_agent(retrieved_chunks: list, query: str) -> dict:
    print(f"\n📝 [Summariser Agent] Compressing {len(retrieved_chunks)} chunks...")

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    raw_context = "\n\n---\n\n".join(retrieved_chunks)

    prompt = f"""You are a context summariser. Compress the retrieved chunks into a concise summary relevant to the question.

QUESTION: {query}

RETRIEVED CHUNKS:
{raw_context}

INSTRUCTIONS:
- Keep only information relevant to the question
- Remove repetition
- Preserve all specific facts, numbers, steps, and technical details
- Output a clean, dense summary (not bullet points)
- Maximum 300 words

SUMMARY:"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )

    summary = response.choices[0].message.content.strip()
    print(f"✅ [Summariser Agent] Context compressed to {len(summary)} characters")

    return {
        "query": query,
        "compressed_context": summary,
        "original_chunks": retrieved_chunks
    }
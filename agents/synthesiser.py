import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

def synthesiser_agent(compressed_context: str, query: str, low_confidence: bool = False) -> dict:
    print(f"\n🧠 [Synthesiser Agent] Generating final answer...")

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    confidence_note = ""
    if low_confidence:
        confidence_note = "\nNOTE: The retrieved context may not be highly relevant to this question."

    prompt = f"""You are an expert research assistant. Answer the user's question using ONLY the context provided.

CONTEXT:
{compressed_context}
{confidence_note}

QUESTION: {query}

STRICT INSTRUCTIONS:
- Answer ONLY from the context above
- If context is insufficient, say exactly: "The document does not contain sufficient information on this topic."
- Do NOT infer, assume, or use outside knowledge
- Be clear, structured, and concise
- Use numbered lists for steps

ANSWER:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800
    )

    answer = response.choices[0].message.content.strip()
    print(f"✅ [Synthesiser Agent] Answer generated ({len(answer)} characters)")

    return {
        "query": query,
        "answer": answer,
        "context_used": compressed_context
    }
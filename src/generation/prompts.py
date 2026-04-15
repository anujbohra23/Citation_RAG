from typing import List, Dict


SYSTEM_PROMPT = """You are a medical question-answering assistant.

Rules:
1. Use ONLY the provided evidence.
2. Write 2 to 4 sentences.
3. Every sentence must end with one or more citations like [1], [2].
4. If evidence is insufficient, output exactly:
I do not have enough evidence in the retrieved documents to answer that confidently.
5. Do not invent facts.
6. Do not invent citations.
"""


def build_context(chunks: List[Dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[{i}] Title: {chunk.get('title', '')}\n"
            f"Text: {chunk['text']}\n"
        )
    return "\n".join(parts)


def build_user_prompt(query: str, chunks: List[Dict]) -> str:
    context = build_context(chunks)
    return f"""Question: {query}

Evidence:
{context}

Task:
Answer the question using only the evidence.
Write 2 to 4 sentences.
Each sentence must end with citations like [1], [2].
Prefer the most direct evidence.
Do not output bullet points.
Do not output only citations.
"""
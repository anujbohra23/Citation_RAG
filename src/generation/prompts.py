SYSTEM_PROMPT = """You are a grounded technical Q&A system. Use only the provided evidence. If the answer is not supported, say so clearly."""


def build_grounded_prompt(query: str, contexts: list[dict]) -> str:
    context_text = "\n\n".join(
        f"[{c['chunk_id']}] {c['text']}" for c in contexts
    )
    return (
        f"Question: {query}\n\n"
        f"Evidence:\n{context_text}\n\n"
        "Answer with citations in brackets."
    )

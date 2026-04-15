from typing import Dict, List


def build_citation_map(chunks: List[Dict]) -> List[Dict]:
    citation_entries = []
    for i, chunk in enumerate(chunks, start=1):
        citation_entries.append(
            {
                "citation_id": i,
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk.get("doc_id"),
                "title": chunk.get("title"),
                "source_path": chunk.get("source_path"),
            }
        )
    return citation_entries


def format_sources(chunks: List[Dict]) -> str:
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        lines.append(
            f"[{i}] {chunk['chunk_id']} | "
            f"title={chunk.get('title', '')} | "
            f"source={chunk.get('source_path', '')}"
        )
    return "\n".join(lines)
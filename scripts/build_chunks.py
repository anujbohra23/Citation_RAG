import json
from pathlib import Path

from src.ingestion.cleaners import clean_document_text
from src.ingestion.load_medquad import load_medquad_documents
from src.chunking.chunker import chunk_corpus
from src.core.types import Document

RAW_DIR = "data/raw/medquad_repo"
OUT_PATH = "data/interim/chunks.jsonl"


def main() -> None:
    docs = load_medquad_documents(RAW_DIR)

    cleaned_docs = []
    for d in docs:
        cleaned_docs.append(
            Document(
                doc_id=d.doc_id,
                title=d.title,
                text=clean_document_text(d.text),
                source_path=d.source_path,
                metadata=d.metadata,
            )
        )

    chunks = chunk_corpus(
        cleaned_docs,
        target_words=220,
        overlap_words=40,
        min_chunk_words=40,
        max_chunk_words=300,
    )

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

    print(f"Loaded {len(docs)} MedQuAD QA documents")
    print(f"Created {len(chunks)} chunks")
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
from sentence_transformers import SentenceTransformer

from src.core.types import Chunk
from src.retrieval.dense_retriever import save_dense_artifact


CHUNKS_PATH = "data/interim/chunks.jsonl"
OUT_PATH = "data/processed/dense_retriever.pkl"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 64


def main() -> None:
    chunks = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            chunks.append(Chunk(**row))

    print(f"Loaded {len(chunks)} chunks")
    print(f"Loading encoder: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    corpus_texts = [chunk.text for chunk in chunks]
    print("Encoding chunk corpus...")
    embeddings = model.encode(
        corpus_texts,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=True,
    )

    embeddings = np.asarray(embeddings, dtype="float32")
    print(f"Embedding shape: {embeddings.shape}")

    print(f"Saving dense retriever artifact to {OUT_PATH}")
    save_dense_artifact(
        out_path=OUT_PATH,
        model_name=MODEL_NAME,
        chunks=chunks,
        embeddings=embeddings,
    )

    print("Done.")


if __name__ == "__main__":
    main()
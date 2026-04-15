import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.core.types import Chunk
from src.core.utils import read_jsonl
from src.retrieval.bm25_retriever import BM25Retriever
from src.indexing.bm25_index import save_bm25_index
from pathlib import Path

CHUNKS_PATH = 'data/interim/chunks.jsonl'
OUT_PATH = 'data/processed/bm25_corpus.pkl'


def main() -> None:
    chunks = [Chunk(**row) for row in read_jsonl(CHUNKS_PATH)]
    retriever = BM25Retriever(chunks)
    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    save_bm25_index(retriever, OUT_PATH)
    print(f'BM25 index built on {len(chunks)} chunks')
    print(f'Saved to {OUT_PATH}')


if __name__ == '__main__':
    main()

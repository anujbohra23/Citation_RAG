import pickle
from src.retrieval.bm25_retriever import BM25Retriever


def save_bm25_index(retriever: BM25Retriever, path: str) -> None:
    with open(path, 'wb') as f:
        pickle.dump(retriever, f)


def load_bm25_index(path: str) -> BM25Retriever:
    with open(path, 'rb') as f:
        return pickle.load(f)

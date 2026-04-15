import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.ingestion.loaders import load_text_documents


def main() -> None:
    docs = load_text_documents('data/raw/docs')
    print(f'Loaded {len(docs)} documents from data/raw/docs')
    for doc in docs[:5]:
        print(f'- {doc.doc_id}: {len(doc.text.split())} words')


if __name__ == '__main__':
    main()

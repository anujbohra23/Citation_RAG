from pathlib import Path
from typing import List
from src.core.types import Document


SUPPORTED_SUFFIXES = {'.txt', '.md'}


def load_text_documents(data_dir: str) -> List[Document]:
    root = Path(data_dir)
    docs: List[Document] = []

    for path in root.rglob('*'):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            text = path.read_text(encoding='utf-8', errors='ignore').strip()
            if not text:
                continue

            docs.append(
                Document(
                    doc_id=path.stem,
                    title=path.stem.replace('_', ' '),
                    text=text,
                    source_path=str(path),
                    metadata={'suffix': path.suffix.lower()},
                )
            )
    return docs

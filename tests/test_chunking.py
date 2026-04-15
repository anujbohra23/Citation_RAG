from src.core.types import Document
from src.chunking.chunker import chunk_document


def test_chunk_document_produces_chunks():
    text = ('Paragraph one with technical content. ' * 40) + '

' + ('Paragraph two. ' * 40)
    doc = Document(doc_id='d1', title='doc', text=text)
    chunks = chunk_document(doc)
    assert len(chunks) >= 1
    assert all(chunk.chunk_id.startswith('d1_chunk_') for chunk in chunks)

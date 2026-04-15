from typing import List
from src.core.types import Document, Chunk
from src.chunking.splitters import split_into_paragraphs, sliding_word_windows


def word_count(text: str) -> int:
    return len(text.split())


def merge_metadata(doc_metadata, chunk_metadata):
    merged = dict(doc_metadata or {})
    merged.update(chunk_metadata or {})
    return merged


def chunk_document(
    doc: Document,
    target_words: int = 260,
    overlap_words: int = 50,
    min_chunk_words: int = 40,
    max_chunk_words: int = 350,
) -> List[Chunk]:
    paragraphs = split_into_paragraphs(doc.text)
    chunks: List[Chunk] = []

    if not paragraphs:
        return chunks

    buffer: List[str] = []
    buffer_word_count = 0
    chunk_num = 0
    offset_cursor = 0

    def flush_buffer() -> None:
        nonlocal buffer, buffer_word_count, chunk_num, offset_cursor, chunks

        if not buffer:
            return

        text = "\n\n".join(buffer).strip()
        wc = word_count(text)

        if wc > max_chunk_words:
            subchunks = sliding_word_windows(text, target_words, overlap_words)
            running_offset = offset_cursor
            for sub_i, subtext in enumerate(subchunks):
                chunk_id = f"{doc.doc_id}_chunk_{chunk_num}"
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        doc_id=doc.doc_id,
                        title=doc.title,
                        text=subtext,
                        source_path=doc.source_path,
                        start_offset=running_offset,
                        end_offset=running_offset + len(subtext),
                        metadata=merge_metadata(
                            doc.metadata,
                            {
                                "word_count": word_count(subtext),
                                "subchunk_index": sub_i,
                            },
                        ),
                    )
                )
                running_offset += len(subtext)
                chunk_num += 1
        elif wc >= min_chunk_words:
            chunk_id = f"{doc.doc_id}_chunk_{chunk_num}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc.doc_id,
                    title=doc.title,
                    text=text,
                    source_path=doc.source_path,
                    start_offset=offset_cursor,
                    end_offset=offset_cursor + len(text),
                    metadata=merge_metadata(
                        doc.metadata,
                        {
                            "word_count": wc,
                        },
                    ),
                )
            )
            chunk_num += 1

        offset_cursor += len(text) + 2
        buffer = []
        buffer_word_count = 0

    for para in paragraphs:
        para_wc = word_count(para)

        if para_wc > max_chunk_words:
            flush_buffer()
            for subtext in sliding_word_windows(para, target_words, overlap_words):
                chunk_id = f"{doc.doc_id}_chunk_{chunk_num}"
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        doc_id=doc.doc_id,
                        title=doc.title,
                        text=subtext,
                        source_path=doc.source_path,
                        start_offset=offset_cursor,
                        end_offset=offset_cursor + len(subtext),
                        metadata=merge_metadata(
                            doc.metadata,
                            {
                                "word_count": word_count(subtext),
                                "from_large_paragraph": True,
                            },
                        ),
                    )
                )
                offset_cursor += len(subtext) + 1
                chunk_num += 1
            continue

        if buffer_word_count + para_wc <= target_words:
            buffer.append(para)
            buffer_word_count += para_wc
        else:
            flush_buffer()
            buffer.append(para)
            buffer_word_count = para_wc

    flush_buffer()
    return chunks


def chunk_corpus(
    docs: List[Document],
    target_words: int = 260,
    overlap_words: int = 50,
    min_chunk_words: int = 40,
    max_chunk_words: int = 350,
) -> List[Chunk]:
    all_chunks: List[Chunk] = []
    for doc in docs:
        all_chunks.extend(
            chunk_document(
                doc,
                target_words=target_words,
                overlap_words=overlap_words,
                min_chunk_words=min_chunk_words,
                max_chunk_words=max_chunk_words,
            )
        )
    return all_chunks
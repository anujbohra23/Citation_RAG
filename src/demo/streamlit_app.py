import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.retrieval.dense_retriever import load_dense_retriever
from src.reranking.cross_encoder_reranker import CrossEncoderReranker
from src.generation.answer_generator import AnswerGenerator
from src.generation.citation_formatter import format_sources


st.set_page_config(
    page_title="CS535 Medical QA Demo",
    page_icon="🩺",
    layout="wide",
)

DENSE_PATH = "data/processed/dense_retriever.pkl"
RERANKER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@st.cache_resource
def load_components():
    dense = load_dense_retriever(DENSE_PATH)
    reranker = CrossEncoderReranker(RERANKER_NAME)
    generator = AnswerGenerator()
    return dense, reranker, generator


def run_pipeline(query: str, first_stage_k: int, final_evidence_k: int):
    dense, reranker, generator = load_components()

    dense_results = dense.search(query, top_k=first_stage_k)
    reranked = reranker.rerank(query, dense_results, top_k=final_evidence_k)
    result = generator.generate(query, reranked)

    return dense_results, reranked, result


def main():
    st.title("🩺 Hybrid RAG Medical QA")
    st.caption(
        "Pipeline: Dense Retriever → Cross-Encoder Reranker → Grounded Answer Composer"
    )

    with st.sidebar:
        st.header("Settings")
        first_stage_k = st.slider("Dense first-stage top-k", 5, 50, 20, 1)
        final_evidence_k = st.slider("Final evidence chunks", 1, 5, 3, 1)
        show_dense = st.checkbox("Show dense first-stage results", value=False)
        show_evidence = st.checkbox("Show evidence text", value=True)

    default_query = "What causes asthma?"
    query = st.text_input("Ask a medical question", value=default_query)

    run_clicked = st.button("Run QA", type="primary")

    if run_clicked and query.strip():
        with st.spinner("Running retrieval, reranking, and grounded answer generation..."):
            dense_results, reranked, result = run_pipeline(
                query=query.strip(),
                first_stage_k=first_stage_k,
                final_evidence_k=final_evidence_k,
            )

        st.subheader("Grounded Answer")
        st.write(result["answer"])

        st.subheader("Sources")
        st.code(format_sources(reranked), language="text")

        if show_evidence:
            st.subheader("Evidence Chunks")
            for i, chunk in enumerate(reranked, start=1):
                with st.expander(f"[{i}] {chunk['chunk_id']}"):
                    st.markdown(f"**Title:** {chunk.get('title', '')}")
                    st.markdown(f"**Score:** rerank rank = {i}")
                    st.write(chunk["text"])

        if show_dense:
            st.subheader("Dense First-Stage Results")
            for item in dense_results[:10]:
                with st.expander(
                    f"Rank {item['rank']} | Score {item['score']:.4f} | {item['chunk_id']}"
                ):
                    st.write(item["text"])

    st.markdown("---")
    st.caption(
        "Current best retrieval stack on MedQuAD: Dense retrieval + cross-encoder reranking."
    )


if __name__ == "__main__":
    main()
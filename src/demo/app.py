import streamlit as st

st.title('CS535 Hybrid RAG Demo')
st.write('Demo scaffold. Connect retrieval and generation pipeline in later milestones.')
query = st.text_input('Ask a technical question')
if query:
    st.info('Retrieval/generation pipeline not yet wired in this starter.')

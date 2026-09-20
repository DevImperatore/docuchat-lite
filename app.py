import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from rag.loader import load_and_split
from rag.embedder import get_embeddings
from rag.retriever import build_vectorstore
from rag.chain import build_rag_chain

load_dotenv()

st.set_page_config(page_title="DocuChat Lite", page_icon=None, layout="centered")
st.title("DocuChat Lite")
st.caption("Chat with any PDF using RAG and Google Gemini")

if not os.getenv("GOOGLE_API_KEY"):
    st.error("GOOGLE_API_KEY is not set. Add it to your .env file.")
    st.stop()

if "chain" not in st.session_state:
    st.session_state.chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None

with st.sidebar:
    st.header("Upload a PDF")
    uploaded = st.file_uploader("Choose a PDF file", type=["pdf"])
    if uploaded and uploaded.name != st.session_state.doc_name:
        with st.spinner("Processing document..."):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded.read())
                    tmp_path = Path(tmp.name)
                docs = load_and_split(tmp_path)
                embeddings = get_embeddings()
                vectorstore = build_vectorstore(docs, embeddings, persist_dir=f"chroma_db/{uploaded.name}")
                st.session_state.chain = build_rag_chain(vectorstore)
                st.session_state.doc_name = uploaded.name
                st.session_state.messages = []
                st.success(f"{len(docs)} chunks indexed")
            except Exception as e:
                st.error(f"Failed to process document: {e}")
            finally:
                tmp_path.unlink(missing_ok=True)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask something about the document...", disabled=st.session_state.chain is None):
    if st.session_state.chain is None:
        st.warning("Upload a PDF first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.chain.invoke({"query": prompt})
                    answer = result["result"]
                    sources = result.get("source_documents", [])
                    st.markdown(answer)
                    if sources:
                        with st.expander("Sources"):
                            for i, doc in enumerate(sources, 1):
                                page = doc.metadata.get("page", "?")
                                st.markdown(f"**Chunk {i}** (page {page})")
                                st.text(doc.page_content[:300] + "...")
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    err_msg = f"Error generating response: {e}"
                    st.error(err_msg)
                    st.session_state.messages.append({"role": "assistant", "content": err_msg})

if st.session_state.chain is None:
    st.info("Upload a PDF in the sidebar to start chatting.")

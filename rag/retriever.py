from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings


def build_vectorstore(docs: list, embeddings: GoogleGenerativeAIEmbeddings, persist_dir: str = "chroma_db") -> Chroma:
    """Build a ChromaDB vectorstore from document chunks."""
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    return vectorstore


def load_vectorstore(embeddings: GoogleGenerativeAIEmbeddings, persist_dir: str = "chroma_db") -> Chroma:
    """Load an existing ChromaDB vectorstore from disk."""
    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)

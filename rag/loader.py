from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.text_cleaner import clean_text


def load_and_split(pdf_path: str | Path, chunk_size: int = 1000, chunk_overlap: int = 150) -> list:
    """Load a PDF and split it into overlapping text chunks."""
    loader = PyPDFLoader(str(pdf_path))
    raw_docs = loader.load()
    for doc in raw_docs:
        doc.page_content = clean_text(doc.page_content)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_documents(raw_docs)

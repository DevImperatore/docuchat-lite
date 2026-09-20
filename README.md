# DocuChat Lite

A lightweight Retrieval-Augmented Generation (RAG) application that lets you chat with any PDF document using Google Gemini.

No GPU required. Runs entirely on CPU. Uses the Gemini free tier (1500 requests/day).

**Live demo:** [Link after deploy to Streamlit Community Cloud]

## How It Works

1. Upload a PDF via the sidebar.
2. The app splits it into chunks, embeds them with Gemini Embeddings, and stores them in a local ChromaDB vectorstore.
3. When you ask a question, the most relevant chunks are retrieved and sent to `gemini-1.5-flash` as context.
4. The model answers using only the document content.

## Tech Stack

- Python 3.12
- Streamlit
- LangChain
- Google Gemini API (`gemini-1.5-flash` + `embedding-001`)
- ChromaDB (local vectorstore)
- PyPDF (PDF parsing)

## Getting Started

```bash
git clone https://github.com/DevImperatore/docuchat-lite.git
cd docuchat-lite
pip install -r requirements.txt
cp .env.example .env
# Add your GOOGLE_API_KEY to .env
streamlit run app.py
```

Get a free Gemini API key at https://aistudio.google.com/apikey

## Project Structure

```
app.py              # Streamlit UI and session state management
rag/
  loader.py        # PDF loading and recursive text splitting
  embedder.py      # Gemini embedding model wrapper
  retriever.py     # ChromaDB vectorstore build and load
  chain.py         # RetrievalQA chain with custom prompt
utils/
  text_cleaner.py  # Whitespace normalization for extracted text
```

## Design Decisions

- **Gemini free tier:** 1500 requests/day is sufficient for demo and personal use. No credit card required.
- **ChromaDB local:** Avoids a hosted vector database. The index is rebuilt per session if the PDF changes.
- **`gemini-1.5-flash`:** Fastest Gemini model with strong instruction following. Adequate for Q&A over documents.
- **Streamlit:** Minimal friction for deploying ML interfaces. No frontend code required.

## License

MIT

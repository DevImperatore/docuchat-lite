from langchain_google_genai import GoogleGenerativeAIEmbeddings


def get_embeddings(model: str = "models/embedding-001") -> GoogleGenerativeAIEmbeddings:
    """Return a Gemini embedding model instance."""
    return GoogleGenerativeAIEmbeddings(model=model)

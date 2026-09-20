from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a precise document assistant. Answer the question using only the provided context. "
        "If the answer is not in the context, say so clearly.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ),
)


def build_rag_chain(vectorstore, model: str = "gemini-1.5-flash", k: int = 4) -> RetrievalQA:
    """Build a RetrievalQA chain with Gemini as the LLM."""
    llm = ChatGoogleGenerativeAI(model=model, temperature=0.1)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": RAG_PROMPT},
        return_source_documents=True,
    )
    return chain

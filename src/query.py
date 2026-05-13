"""Query pipeline: retrieve -> augment -> generate."""
from pathlib import Path
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate


TOP_K = 4
LLM_MODEL = "gpt-4o-mini"
TEMPERATURE = 0
EMBEDDING_MODEL = "text-embedding-3-small"


PROMPT_TEMPLATE = """You are a financial research assistant answering questions based on SEC 10-K filings.

Answer the question using ONLY the information in the provided context. If the context does not contain enough information to answer, say "I don't have enough information in the provided filings to answer that."

Always cite which company\'s filing your information comes from. When facts come from multiple companies, cite each one.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


def load_vectorstore(index_path: str = "data/faiss_index"):
    """Load a persisted FAISS index from disk."""
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True
    )


def format_context(retrieved_docs):
    """Format retrieved chunks with clear source attribution."""
    parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        ticker = doc.metadata.get("ticker", "?")
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[Source {i}: {ticker} ({source})]\n{doc.page_content}")
    return "\n\n".join(parts)


def answer(question: str, vectorstore=None, verbose: bool = True):
    """Full RAG pipeline: retrieve, augment, generate."""
    if vectorstore is None:
        vectorstore = load_vectorstore()
    
    retrieved = vectorstore.similarity_search(question, k=TOP_K)
    context = format_context(retrieved)
    
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    llm = ChatOpenAI(model=LLM_MODEL, temperature=TEMPERATURE)
    chain = prompt | llm
    result = chain.invoke({"context": context, "question": question})
    
    if verbose:
        print(f"Q: {question}\n")
        print(f"A: {result.content}\n")
        sources = sorted(set(d.metadata.get("ticker", "?") for d in retrieved))
        print(f"Retrieved from: {sources}")
    
    return result.content, retrieved

"""Ingestion pipeline: load -> chunk -> embed -> store in FAISS."""
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from .loader import load_all_filings


CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "text-embedding-3-small"


def chunk_documents(docs):
    """Recursive splitter respects paragraph -> sentence -> word boundaries."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def build_index(chunks, save_path: str = "data/faiss_index"):
    """Embed chunks and store in FAISS index, persisted to disk."""
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    Path(save_path).mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(save_path)
    
    return vectorstore


def ingest(data_dir: str = "data", index_path: str = "data/faiss_index"):
    """Full ingestion: load filings, chunk, embed, save FAISS index."""
    print(f"Loading filings from {data_dir}...")
    docs = load_all_filings(data_dir)
    print(f"Loaded {len(docs)} documents")
    
    print("Chunking...")
    chunks = chunk_documents(docs)
    print(f"Produced {len(chunks):,} chunks")
    
    print(f"Embedding with {EMBEDDING_MODEL} (this takes 5-10 min for large corpora)...")
    vectorstore = build_index(chunks, index_path)
    print(f"FAISS index built: {vectorstore.index.ntotal:,} vectors, saved to {index_path}/")
    
    return vectorstore

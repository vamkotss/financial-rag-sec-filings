# Financial RAG System over SEC 10-K Filings

A retrieval-augmented generation system that answers natural-language questions about public companies using their SEC 10-K annual reports as the knowledge source. Built with LangChain, FAISS, and OpenAI.

## Business problem

Equity research analysts at investment banks, hedge funds, and asset managers spend hours every day searching SEC filings, earnings transcripts, and analyst notes for specific information. Questions like "What does Tesla say about Robotaxi regulations?" or "How does Nvidia describe AI chip demand?" require precise, cited answers from actual source documents. A general-purpose LLM cannot answer these because it never saw the specific filings and may have stale data. You cannot paste a 300-page 10-K into a prompt at any reasonable cost. RAG retrieves relevant chunks at query time and grounds the answer in actual source material with citations.

## Architecture

```
Ingestion phase (runs once):
  SEC HTML filings
      |
      v
  BeautifulSoup (lxml parser, strip tags, preserve text)
      |
      v
  RecursiveCharacterTextSplitter (500 chars, 50 overlap)
      |
      v
  OpenAI text-embedding-3-small (1536 dims)
      |
      v
  FAISS index (persisted to disk)

Query phase (runs per question):
  User question
      |
      v
  Embed question with same model
      |
      v
  FAISS cosine similarity search (top-4)
      |
      v
  Format chunks with source attribution
      |
      v
  Grounded prompt template (refuse to hallucinate)
      |
      v
  gpt-4o-mini (temperature=0)
      |
      v
  Cited answer + source documents
```

## Corpus

Four real SEC 10-K filings:

- Apple Inc (AAPL) - FY2024
- Meta Platforms (META) - FY2025
- Nvidia Corporation (NVDA) - FY2025
- Tesla Inc (TSLA) - FY2025

## Key design decisions

**Chunk size 500 with 50 overlap**: tested 1000 first, but larger chunks diluted retrieval precision when answers were specific phrases (regulatory language, product names, segment definitions). 500 captures enough context without dilution. 10% overlap handles cases where an answer spans chunk boundaries.

**Recursive character splitter**: tries paragraph breaks first, then sentence, then word, then character fallback. Respects semantic boundaries instead of cutting mid-sentence the way fixed-character splitters do.

**text-embedding-3-small over -large**: 5x cheaper, 1536 vs 3072 dimensions, slightly lower quality. For RAG over text-dense documents, the bottleneck is rarely embedding quality — it is chunking strategy and retrieval logic. Worth upgrading only if eval shows embeddings are the limiting factor.

**Top-k=4**: balance between coverage and noise. Higher k brings more relevant context but also more irrelevant chunks that dilute the prompt and increase cost. Lower k risks missing the chunk that contains the answer.

**Temperature 0**: factual Q&A should be deterministic. Same question, same answer across runs.

**Explicit refusal instruction**: the prompt tells the model to say "I don\'t have enough information" if the context does not cover the answer. Verified by asking off-topic questions and confirming the model refuses to hallucinate.

**Source attribution in prompt**: every chunk is prefixed with its ticker and source filename. The model cites which company each fact comes from, creating an audit trail.

## Evaluation

Built a basic eval harness with 5 hand-crafted Q&A pairs whose answers are known to exist in the filings. Scored by keyword presence in the generated answer.

**Average score: 0.80**

Detailed results: 2 questions scored 1.00, 3 questions scored 0.67. The 0.67 scores were not factually wrong — they were correct answers that used different vocabulary than my expected keywords. For example, Nvidia\'s data center answer described "accelerating compute-intensive workloads" instead of using the literal word "GPU." This exposes the core limitation of keyword-match evaluation: it cannot recognize semantic equivalence.

Production-grade evaluation would use:
- **RAGAs framework**: faithfulness, answer relevance, context precision metrics
- **LLM-as-judge**: strong model scores answers against rubrics
- **Human eval on a sample**: gold standard for high-stakes outputs

## Known limitations

**Multi-entity questions fail**: asking "How do Nvidia AND Meta describe their AI investments?" returned 4 chunks all from Nvidia because their AI infrastructure language dominates the embedding space. Single-vector similarity search cannot guarantee coverage across multiple entities mentioned in one question.

Production fix: **query decomposition** (split multi-entity questions into per-entity sub-questions, retrieve separately, synthesize) or **per-source retrieval** (k chunks per filing, guaranteed coverage).

## Stack

- Python 3.11+
- LangChain (orchestration, loaders, splitters)
- BeautifulSoup + lxml (HTML parsing)
- OpenAI text-embedding-3-small (embeddings) + gpt-4o-mini (generation)
- FAISS (vector similarity search)

## Usage

```python
from finrag.src import ingest, answer

# Run once to build the index
ingest(data_dir="data", index_path="data/faiss_index")

# Then query
answer("What does Tesla say about Robotaxi regulations?")
```

## Cost

- One-time embedding cost for 4 filings (~20K chunks): ~$0.40
- Per query: ~$0.001 (one embedding call + one gpt-4o-mini call)

## What I would improve next

- **Query decomposition** for multi-entity questions (described above)
- **Hybrid search**: combine BM25 keyword search with vector search to catch rare-term queries (specific dollar amounts, product names)
- **Reranking** with Cohere Rerank or a cross-encoder on top of initial retrieval
- **Semantic chunking** that uses embeddings to find natural breakpoints instead of fixed character counts
- **Query rewriting**: small LLM call to rephrase user questions into terms that better match document language
- **Production eval**: replace keyword matching with RAGAs and LLM-as-judge
- **Migrate to AWS Bedrock Knowledge Bases** for production deployment with HIPAA/SOC2 compliance posture

## License

MIT
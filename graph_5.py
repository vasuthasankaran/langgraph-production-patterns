"""
Pattern 03: Streaming RAG with Azure AI Search + Citation Tracking
==================================================================
Production use-case: Enterprise document Q&A where every claim in the
answer must be traceable to a source document + chunk. Streams tokens
to the caller while building a citation map in parallel.

Key concepts:
- Azure AI Search hybrid retrieval (vector + keyword BM25)
- Semantic reranking with Azure's built-in reranker
- Citation tracking: each sentence tagged to source doc + page
- Streaming via astream_events (not just astream)
- Query rewriting for multi-hop questions
- Confidence scoring to decide when to say "I don't know"
"""

import asyncio
import json
from typing import AsyncIterator, Annotated, Any
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.documents import Document
from pydantic import BaseModel, Field

try:
    from azure.search.documents.aio import SearchClient
    from azure.search.documents.models import VectorizedQuery, QueryType
    from azure.core.credentials import AzureKeyCredential
    AZURE_SEARCH_AVAILABLE = True
except ImportError:
    AZURE_SEARCH_AVAILABLE = False


# ── Models ────────────────────────────────────────────────────────────────────

class Citation(BaseModel):
    citation_id: str
    source_document: str
    page_number: int | None = None
    chunk_id: str
    relevance_score: float
    excerpt: str       # the retrieved chunk text (truncated for display)


class RAGState(BaseModel):
    query: str = ""
    rewritten_query: str = ""
    retrieved_chunks: list[dict[str, Any]] = []
    citations: list[Citation] = []
    answer: str = ""
    answer_with_citations: str = ""
    confidence: float = 0.0
    low_confidence_fallback: bool = False
    session_id: str = Field(default_factory=lambda: str(uuid4()))


# ── Azure Search client ───────────────────────────────────────────────────────

def get_search_client(index_name: str) -> "SearchClient":
    import os
    return SearchClient(
        endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
        index_name=index_name,
        credential=AzureKeyCredential(os.environ["AZURE_SEARCH_API_KEY"]),
    )


def get_embeddings() -> AzureOpenAIEmbeddings:
    return AzureOpenAIEmbeddings(
        azure_deployment="text-embedding-3-large",
        # AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY from env
    )


def get_llm(streaming: bool = False) -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_deployment="gpt-4o",
        api_version="2024-10-21",
        temperature=0,
        streaming=streaming,
    )


# ── Nodes ─────────────────────────────────────────────────────────────────────

async def rewrite_query(state: RAGState) -> dict:
    """
    Rewrites the user query to be retrieval-optimised.
    Handles multi-hop: 'What changed between v1 and v2?' → two sub-queries.
    For simplicity this pattern returns a single rewritten query.
    """
    llm = get_llm()
    response = await llm.ainvoke([
        SystemMessage(content="""Rewrite the user's question as an optimal search query for a vector + keyword index.
Remove conversational filler. Expand acronyms. Output only the rewritten query, nothing else."""),
        HumanMessage(content=state.query),
    ])
    rewritten = response.content.strip()
    print(f"  [QueryRewrite] '{state.query}' → '{rewritten}'")
    return {"rewritten_query": rewritten}


async def retrieve_chunks(state: RAGState) -> dict:
    """
    Hybrid retrieval: vector similarity + BM25 keyword, then semantic rerank.
    Falls back to mock data if Azure Search isn't configured.
    """
    if not AZURE_SEARCH_AVAILABLE or not _azure_configured():
        return {"retrieved_chunks": _mock_chunks(state.rewritten_query)}

    embeddings = get_embeddings()
    query_vector = await embeddings.aembed_query(state.rewritten_query)

    async with get_search_client("enterprise-docs") as client:
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=50,
            fields="content_vector",
        )
        results = await client.search(
            search_text=state.rewritten_query,
            vector_queries=[vector_query],
            query_type=QueryType.SEMANTIC,
            semantic_configuration_name="default",
            top=8,
            select=["id", "content", "source_document", "page_number", "chunk_id"],
        )
        chunks = []
        async for r in results:
            chunks.append({
                "id": r["id"],
                "content": r["content"],
                "source_document": r["source_document"],
                "page_number": r.get("page_number"),
                "chunk_id": r.get("chunk_id", r["id"]),
                "score": r["@search.reranker_score"] or r["@search.score"],
            })

    print(f"  [Retrieval] Found {len(chunks)} chunks")
    return {"retrieved_chunks": chunks}


async def score_confidence(state: RAGState) -> dict:
    """
    Estimate retrieval confidence before generating.
    If top chunk score is low, we'll signal a fallback response.
    """
    if not state.retrieved_chunks:
        return {"confidence": 0.0, "low_confidence_fallback": True}

    top_score = max(c.get("score", 0) for c in state.retrieved_chunks)
    # Azure semantic reranker scores 0–4; normalize to 0–1
    normalized = min(top_score / 4.0, 1.0) if top_score > 1 else top_score
    fallback = normalized < 0.35

    print(f"  [Confidence] top_score={top_score:.3f}, normalized={normalized:.3f}, fallback={fallback}")
    return {"confidence": normalized, "low_confidence_fallback": fallback}


async def generate_answer(state: RAGState) -> dict:
    """
    Generates a grounded answer. Instructs the LLM to annotate every
    factual claim with [CITE:chunk_id] markers which we post-process
    into a proper citation map.
    """
    if state.low_confidence_fallback:
        answer = ("I couldn't find sufficiently relevant information in the knowledge base "
                  "to answer this question confidently. Please rephrase or check the source documents.")
        return {"answer": answer, "answer_with_citations": answer, "citations": []}

    context_blocks = []
    for i, chunk in enumerate(state.retrieved_chunks[:6]):
        context_blocks.append(
            f"[CHUNK:{chunk['chunk_id']}] (Source: {chunk['source_document']}, "
            f"Page: {chunk.get('page_number', 'N/A')})\n{chunk['content']}"
        )
    context = "\n\n---\n\n".join(context_blocks)

    llm = get_llm()
    response = await llm.ainvoke([
        SystemMessage(content=f"""Answer the question using ONLY the provided context.
After every factual sentence, add [CITE:chunk_id] using the exact chunk IDs from the context.
If the context doesn't support a claim, do not make it.
If you cannot answer, say so explicitly.

Context:
{context}"""),
        HumanMessage(content=state.query),
    ])

    raw_answer = response.content
    citations, clean_answer = _extract_citations(raw_answer, state.retrieved_chunks)
    print(f"  [Generate] {len(citations)} citations extracted")

    return {
        "answer": clean_answer,
        "answer_with_citations": raw_answer,
        "citations": citations,
    }


# ── Streaming variant ─────────────────────────────────────────────────────────

async def stream_answer(query: str, index_name: str = "enterprise-docs") -> AsyncIterator[str]:
    """
    Streaming entry point — yields tokens as they arrive.
    Citations are emitted as a final JSON block after streaming completes.

    Usage:
        async for chunk in stream_answer("What is our supplier onboarding SLA?"):
            print(chunk, end="", flush=True)
    """
    graph = build_graph()
    initial = RAGState(query=query)

    # Run retrieval nodes synchronously (non-streaming)
    state_after_retrieval = await graph.ainvoke(
        initial,
        config={"run_name": "rag_retrieve_phase"}
    )

    # Stream only the generation step
    llm = get_llm(streaming=True)
    context_blocks = [
        f"[CHUNK:{c['chunk_id']}] {c['content']}"
        for c in state_after_retrieval.retrieved_chunks[:6]
    ]

    async for chunk in llm.astream([
        SystemMessage(content=f"Answer using context. Cite with [CITE:chunk_id].\n\n{'---'.join(context_blocks)}"),
        HumanMessage(content=query),
    ]):
        yield chunk.content

    yield f"\n\n<!-- citations: {json.dumps([c.model_dump() for c in state_after_retrieval.citations])} -->"


# ── Graph ─────────────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(RAGState)
    g.add_node("rewrite_query", rewrite_query)
    g.add_node("retrieve_chunks", retrieve_chunks)
    g.add_node("score_confidence", score_confidence)
    g.add_node("generate_answer", generate_answer)

    g.add_edge(START, "rewrite_query")
    g.add_edge("rewrite_query", "retrieve_chunks")
    g.add_edge("retrieve_chunks", "score_confidence")
    g.add_edge("score_confidence", "generate_answer")
    g.add_edge("generate_answer", END)

    return g.compile()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _azure_configured() -> bool:
    import os
    return bool(os.environ.get("AZURE_SEARCH_ENDPOINT") and os.environ.get("AZURE_SEARCH_API_KEY"))


def _mock_chunks(query: str) -> list[dict]:
    return [
        {
            "id": "chunk-001", "chunk_id": "chunk-001",
            "content": f"Mock chunk 1 relevant to: {query}. This contains enterprise procurement policy details.",
            "source_document": "procurement_policy_v3.pdf",
            "page_number": 12,
            "score": 3.2,
        },
        {
            "id": "chunk-002", "chunk_id": "chunk-002",
            "content": "Supplier onboarding SLA is defined as 10 business days from contract signature.",
            "source_document": "supplier_handbook_2024.pdf",
            "page_number": 8,
            "score": 2.9,
        },
    ]


def _extract_citations(raw_answer: str, chunks: list[dict]) -> tuple[list[Citation], str]:
    import re
    chunk_map = {c["chunk_id"]: c for c in chunks}
    citations = []
    seen = set()

    for match in re.finditer(r'\[CITE:([^\]]+)\]', raw_answer):
        cid = match.group(1)
        if cid in chunk_map and cid not in seen:
            seen.add(cid)
            c = chunk_map[cid]
            citations.append(Citation(
                citation_id=f"[{len(citations)+1}]",
                source_document=c["source_document"],
                page_number=c.get("page_number"),
                chunk_id=cid,
                relevance_score=c.get("score", 0.0),
                excerpt=c["content"][:200],
            ))

    clean = re.sub(r'\s*\[CITE:[^\]]+\]', '', raw_answer).strip()
    return citations, clean


async def demo():
    print("=== Pattern 03: Streaming RAG + Citation Tracking ===\n")
    graph = build_graph()
    result = await graph.ainvoke(RAGState(query="What is the supplier onboarding SLA?"))
    print(f"  Answer: {result.answer}")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Citations ({len(result.citations)}):")
    for c in result.citations:
        print(f"    {c.citation_id} {c.source_document} p.{c.page_number} — {c.excerpt[:80]}...")


if __name__ == "__main__":
    asyncio.run(demo())

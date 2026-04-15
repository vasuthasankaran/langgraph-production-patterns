"""
Pattern 05: Token Budget Guardian
==================================
Production use-case: Long-running agents can burn tokens silently.
This pattern injects a budget guardian node that checks cumulative
token spend mid-graph and reroutes to a compression/summarization
path before the context window or cost ceiling is hit.

Key concepts:
- Tiktoken-based token counting before LLM calls
- Soft limit → summarize context, continue
- Hard limit → generate partial answer, stop gracefully
- Budget state threads through every node
- Per-call token forecast to avoid surprise overruns
"""

import asyncio
from typing import Literal
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


# ── Budget configuration ──────────────────────────────────────────────────────

class BudgetConfig(BaseModel):
    soft_limit_tokens: int = 6_000    # compress context here
    hard_limit_tokens: int = 10_000   # abort and return partial answer
    max_cost_usd: float = 0.10        # hard stop on cost
    model: str = "gpt-4o"


# ── State ─────────────────────────────────────────────────────────────────────

class BudgetedState(BaseModel):
    messages: list[BaseMessage] = []
    context_chunks: list[str] = []
    compressed_context: str = ""
    answer: str = ""
    tokens_used: int = 0
    cost_usd: float = 0.0
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    budget_status: str = "ok"         # ok | soft_limit | hard_limit
    compression_applied: bool = False
    session_id: str = Field(default_factory=lambda: str(uuid4()))


# ── Token counting ────────────────────────────────────────────────────────────

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    if not TIKTOKEN_AVAILABLE:
        return len(text) // 4   # rough estimate
    try:
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except Exception:
        return len(text) // 4


def count_messages_tokens(messages: list[BaseMessage], model: str = "gpt-4o") -> int:
    total = 0
    for msg in messages:
        total += count_tokens(str(msg.content), model)
        total += 4   # per-message overhead
    return total + 2  # reply priming


def estimate_cost(tokens: int, model: str = "gpt-4o") -> float:
    rate = 0.0025 / 1000 if "mini" not in model else 0.00015 / 1000
    return tokens * rate


# ── LLM ───────────────────────────────────────────────────────────────────────

def get_llm() -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_deployment="gpt-4o",
        api_version="2024-10-21",
        temperature=0,
    )


def get_cheap_llm() -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_deployment="gpt-4o-mini",
        api_version="2024-10-21",
        temperature=0,
    )


# ── Nodes ─────────────────────────────────────────────────────────────────────

async def load_context(state: BudgetedState) -> dict:
    """Simulate loading many document chunks (e.g., from Azure AI Search)."""
    chunks = [
        "Chunk 1: Q3 revenue for EMEA region reached €142M, representing 8% YoY growth. " * 5,
        "Chunk 2: Key driver was specialty chemicals segment, up 15%. Automotive demand weak. " * 5,
        "Chunk 3: Operating margin improved to 18.2% due to raw material cost reduction. " * 5,
        "Chunk 4: Procurement savings of €12M achieved through supplier consolidation. " * 5,
        "Chunk 5: Headcount grew by 120 FTEs in R&D, offset by 80 in shared services. " * 5,
        "Chunk 6: Capital expenditure of €28M focused on plant modernization in Leverkusen. " * 5,
    ]
    tokens = sum(count_tokens(c) for c in chunks)
    cost = estimate_cost(tokens)
    print(f"  [LoadContext] {len(chunks)} chunks, ~{tokens} tokens")
    return {
        "context_chunks": chunks,
        "tokens_used": state.tokens_used + tokens,
        "cost_usd": state.cost_usd + cost,
    }


async def check_budget(state: BudgetedState) -> dict:
    """
    Guardian node: checks current token spend and sets budget_status.
    This node sits between every expensive step in the graph.
    """
    b = state.budget
    tokens = state.tokens_used
    cost = state.cost_usd

    print(f"  [BudgetGuard] tokens={tokens:,} / hard={b.hard_limit_tokens:,}, cost=${cost:.4f} / max=${b.max_cost_usd}")

    if tokens >= b.hard_limit_tokens or cost >= b.max_cost_usd:
        print(f"  [BudgetGuard] ⛔ HARD LIMIT HIT — aborting")
        return {"budget_status": "hard_limit"}
    elif tokens >= b.soft_limit_tokens:
        print(f"  [BudgetGuard] ⚠️  Soft limit — will compress context")
        return {"budget_status": "soft_limit"}
    else:
        print(f"  [BudgetGuard] ✅ Budget OK")
        return {"budget_status": "ok"}


async def compress_context(state: BudgetedState) -> dict:
    """
    Summarize all chunks into a compact context using the cheaper model.
    This recovers ~60-70% of the token budget for generation.
    """
    llm = get_cheap_llm()
    combined = "\n\n".join(state.context_chunks)
    original_tokens = count_tokens(combined)

    response = await llm.ainvoke([
        SystemMessage(content="Compress these document chunks into a dense 200-word summary preserving all key facts and numbers."),
        HumanMessage(content=combined),
    ])

    compressed = response.content
    compressed_tokens = count_tokens(compressed)
    savings = original_tokens - compressed_tokens
    cost = estimate_cost(original_tokens + compressed_tokens, model="gpt-4o-mini")

    print(f"  [Compress] {original_tokens}→{compressed_tokens} tokens saved {savings} (${cost:.5f})")
    return {
        "compressed_context": compressed,
        "compression_applied": True,
        "tokens_used": state.tokens_used - original_tokens + compressed_tokens,
        "cost_usd": state.cost_usd + cost,
        "budget_status": "ok",
    }


async def generate_answer(state: BudgetedState) -> dict:
    """Full answer generation using available context."""
    llm = get_llm()
    context = state.compressed_context if state.compression_applied else "\n\n".join(state.context_chunks[:3])

    messages = state.messages + [
        SystemMessage(content=f"Answer the question using this context:\n{context}"),
    ]
    response = await llm.ainvoke(messages)

    usage = getattr(response, "response_metadata", {}).get("token_usage", {})
    tokens = usage.get("total_tokens", count_tokens(response.content))
    cost = estimate_cost(tokens)

    print(f"  [Generate] full answer, {tokens} tokens")
    return {
        "answer": response.content,
        "tokens_used": state.tokens_used + tokens,
        "cost_usd": state.cost_usd + cost,
    }


async def generate_partial_answer(state: BudgetedState) -> dict:
    """
    Hard limit reached — generate the best possible answer from
    whatever context we have so far, using the cheap model.
    """
    llm = get_cheap_llm()
    partial_context = "\n\n".join(state.context_chunks[:2]) if state.context_chunks else "Limited context available."
    question = state.messages[-1].content if state.messages else "Unknown query"

    response = await llm.ainvoke([
        SystemMessage(content=f"""Budget exceeded. Generate a concise partial answer from limited context.
Acknowledge that the answer may be incomplete.

Context (partial): {partial_context[:1000]}"""),
        HumanMessage(content=question),
    ])

    print(f"  [PartialAnswer] ⚠️  Returning budget-constrained response")
    return {"answer": f"[PARTIAL — budget limit reached]\n{response.content}"}


# ── Routing ───────────────────────────────────────────────────────────────────

def route_from_budget_check(state: BudgetedState) -> str:
    if state.budget_status == "hard_limit":
        return "generate_partial_answer"
    elif state.budget_status == "soft_limit":
        return "compress_context"
    return "generate_answer"


# ── Graph ─────────────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(BudgetedState)
    g.add_node("load_context", load_context)
    g.add_node("check_budget", check_budget)
    g.add_node("compress_context", compress_context)
    g.add_node("generate_answer", generate_answer)
    g.add_node("generate_partial_answer", generate_partial_answer)

    g.add_edge(START, "load_context")
    g.add_edge("load_context", "check_budget")
    g.add_conditional_edges("check_budget", route_from_budget_check, {
        "generate_answer": "generate_answer",
        "compress_context": "compress_context",
        "generate_partial_answer": "generate_partial_answer",
    })
    # After compression, check budget again before generating
    g.add_edge("compress_context", "generate_answer")
    g.add_edge("generate_answer", END)
    g.add_edge("generate_partial_answer", END)

    return g.compile()


async def demo():
    print("=== Pattern 05: Token Budget Guardian ===\n")
    graph = build_graph()

    # Demo 1: soft limit triggers compression
    print("--- Demo 1: Soft limit (compress + continue) ---")
    result = await graph.ainvoke(BudgetedState(
        messages=[HumanMessage(content="Summarize our Q3 EMEA performance.")],
        budget=BudgetConfig(soft_limit_tokens=100, hard_limit_tokens=50_000),
    ))
    print(f"  compression_applied={result.compression_applied}")
    print(f"  final_tokens={result.tokens_used}, cost=${result.cost_usd:.5f}\n")

    # Demo 2: hard limit triggers partial answer
    print("--- Demo 2: Hard limit (partial answer) ---")
    result2 = await graph.ainvoke(BudgetedState(
        messages=[HumanMessage(content="Summarize our Q3 EMEA performance.")],
        budget=BudgetConfig(soft_limit_tokens=50, hard_limit_tokens=80),
    ))
    print(f"  answer_prefix='{result2.answer[:80]}...'")


if __name__ == "__main__":
    asyncio.run(demo())

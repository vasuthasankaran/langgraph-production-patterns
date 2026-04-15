"""
Pattern 04: LangFuse Production Observability
=============================================
Production use-case: Full trace coverage for a multi-step agent —
every LLM call, retrieval, tool use, and routing decision is captured
with token counts, latencies, costs, and evaluation scores.

Key concepts:
- LangFuse CallbackHandler wired into every LLM and graph call
- Custom span creation for non-LLM steps (retrieval, tool calls)
- Per-session cost accumulation
- LLM-as-judge eval scores written back to LangFuse
- Structured metadata tagging (user_id, session_id, environment)
- Score flushing on graph completion
"""

import asyncio
import os
import time
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

try:
    from langfuse import Langfuse
    from langfuse.callback import CallbackHandler as LangfuseCallbackHandler
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    print("  [warn] langfuse not installed — observability disabled. pip install langfuse")


# ── Observability client ──────────────────────────────────────────────────────

def get_langfuse() -> "Langfuse | None":
    if not LANGFUSE_AVAILABLE:
        return None
    return Langfuse(
        public_key=os.environ.get("LANGFUSE_PUBLIC_KEY", ""),
        secret_key=os.environ.get("LANGFUSE_SECRET_KEY", ""),
        host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
    )


def get_langfuse_handler(session_id: str, user_id: str, tags: list[str] | None = None) -> Any:
    """Returns a LangChain callback handler or a no-op if LangFuse isn't configured."""
    if not LANGFUSE_AVAILABLE:
        return None
    return LangfuseCallbackHandler(
        public_key=os.environ.get("LANGFUSE_PUBLIC_KEY", ""),
        secret_key=os.environ.get("LANGFUSE_SECRET_KEY", ""),
        host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        session_id=session_id,
        user_id=user_id,
        tags=tags or [],
        metadata={"environment": os.environ.get("APP_ENV", "development")},
    )


# ── State ─────────────────────────────────────────────────────────────────────

class ObservableState(BaseModel):
    query: str = ""
    context: str = ""
    answer: str = ""
    eval_score: float | None = None
    eval_reasoning: str = ""
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str = "anonymous"
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    latencies_ms: dict[str, float] = {}


# ── Cost tracking ─────────────────────────────────────────────────────────────

GPT4O_COST_PER_1K = {"input": 0.0025, "output": 0.010}
GPT4O_MINI_COST_PER_1K = {"input": 0.00015, "output": 0.0006}

def estimate_cost(input_tokens: int, output_tokens: int, model: str = "gpt-4o") -> float:
    rates = GPT4O_COST_PER_1K if "mini" not in model else GPT4O_MINI_COST_PER_1K
    return (input_tokens / 1000 * rates["input"]) + (output_tokens / 1000 * rates["output"])


# ── LLM factory (with observability) ─────────────────────────────────────────

def get_llm(session_id: str, user_id: str, step_name: str) -> AzureChatOpenAI:
    handler = get_langfuse_handler(session_id, user_id, tags=[step_name])
    kwargs: dict[str, Any] = {
        "azure_deployment": "gpt-4o",
        "api_version": "2024-10-21",
        "temperature": 0,
    }
    if handler:
        kwargs["callbacks"] = [handler]
    return AzureChatOpenAI(**kwargs)


# ── Nodes ─────────────────────────────────────────────────────────────────────

async def retrieve(state: ObservableState) -> dict:
    lf = get_langfuse()
    t0 = time.monotonic()

    # Manual span for non-LLM steps (retrieval, DB calls, tool use)
    span = None
    if lf:
        trace = lf.trace(
            id=state.session_id,
            name="rag_pipeline",
            user_id=state.user_id,
            input={"query": state.query},
        )
        span = trace.span(
            name="azure_search_retrieval",
            input={"query": state.query},
            metadata={"index": "enterprise-docs", "top_k": 5},
        )

    # Simulate retrieval (replace with actual Azure AI Search call)
    await asyncio.sleep(0.05)
    context = f"Simulated retrieved context for: {state.query}\nFact: Our Q3 EMEA revenue was €142M, up 8% YoY."

    latency = (time.monotonic() - t0) * 1000
    if span:
        span.end(output={"chunks_retrieved": 2, "top_score": 3.1})

    print(f"  [Retrieval] done in {latency:.0f}ms")
    return {"context": context, "latencies_ms": {**state.latencies_ms, "retrieval_ms": latency}}


async def generate(state: ObservableState) -> dict:
    t0 = time.monotonic()
    llm = get_llm(state.session_id, state.user_id, step_name="generation")

    response = await llm.ainvoke([
        SystemMessage(content=f"Answer based on context:\n{state.context}"),
        HumanMessage(content=state.query),
    ])

    # Extract token usage from response metadata
    usage = getattr(response, "response_metadata", {}).get("token_usage", {})
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)
    cost = estimate_cost(input_tokens, output_tokens)
    latency = (time.monotonic() - t0) * 1000

    print(f"  [Generate] {input_tokens}→{output_tokens} tokens, ${cost:.5f}, {latency:.0f}ms")
    return {
        "answer": response.content,
        "total_tokens": state.total_tokens + input_tokens + output_tokens,
        "total_cost_usd": state.total_cost_usd + cost,
        "latencies_ms": {**state.latencies_ms, "generation_ms": latency},
    }


async def evaluate_answer(state: ObservableState) -> dict:
    """
    LLM-as-judge: scores the answer for faithfulness and relevance.
    Writes scores back to LangFuse for dashboard tracking.
    """
    t0 = time.monotonic()
    llm = get_llm(state.session_id, state.user_id, step_name="llm_judge")

    response = await llm.ainvoke([
        SystemMessage(content="""You are an evaluation judge. Score the answer for:
1. Faithfulness (0–1): Is the answer supported by the context?
2. Relevance (0–1): Does it answer the question?

Respond ONLY as JSON: {"faithfulness": 0.0-1.0, "relevance": 0.0-1.0, "reasoning": "..."}"""),
        HumanMessage(content=f"Question: {state.query}\nContext: {state.context}\nAnswer: {state.answer}"),
    ])

    import json, re
    match = re.search(r'\{.*\}', response.content, re.DOTALL)
    scores = json.loads(match.group()) if match else {"faithfulness": 0.5, "relevance": 0.5, "reasoning": "parse error"}

    composite = (scores["faithfulness"] + scores["relevance"]) / 2
    latency = (time.monotonic() - t0) * 1000

    # Write eval scores back to LangFuse
    lf = get_langfuse()
    if lf:
        lf.score(
            trace_id=state.session_id,
            name="faithfulness",
            value=scores["faithfulness"],
            comment=scores.get("reasoning", ""),
        )
        lf.score(
            trace_id=state.session_id,
            name="relevance",
            value=scores["relevance"],
        )
        lf.flush()

    print(f"  [Eval] faithfulness={scores['faithfulness']:.2f}, relevance={scores['relevance']:.2f}, {latency:.0f}ms")
    return {
        "eval_score": composite,
        "eval_reasoning": scores.get("reasoning", ""),
        "latencies_ms": {**state.latencies_ms, "eval_ms": latency},
    }


async def finalize(state: ObservableState) -> dict:
    """Write final trace metadata to LangFuse and log summary."""
    lf = get_langfuse()
    if lf:
        trace = lf.trace(
            id=state.session_id,
            output={
                "answer": state.answer,
                "eval_score": state.eval_score,
            },
            metadata={
                "total_tokens": state.total_tokens,
                "total_cost_usd": round(state.total_cost_usd, 6),
                "total_latency_ms": sum(state.latencies_ms.values()),
            },
        )
        lf.flush()

    print(f"\n  [Summary] tokens={state.total_tokens}, cost=${state.total_cost_usd:.5f}, "
          f"eval={state.eval_score:.2f if state.eval_score else 'N/A'}")
    return {}


# ── Graph ─────────────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(ObservableState)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)
    g.add_node("evaluate_answer", evaluate_answer)
    g.add_node("finalize", finalize)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "evaluate_answer")
    g.add_edge("evaluate_answer", "finalize")
    g.add_edge("finalize", END)

    return g.compile()


async def demo():
    print("=== Pattern 04: LangFuse Observability ===\n")
    graph = build_graph()
    result = await graph.ainvoke(ObservableState(
        query="What was our Q3 EMEA revenue performance?",
        user_id="demo-user",
    ))
    print(f"\n  Answer: {result.answer[:150]}...")
    print(f"  Eval score: {result.eval_score}")


if __name__ == "__main__":
    asyncio.run(demo())

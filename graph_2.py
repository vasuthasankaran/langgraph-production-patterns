"""
Pattern 06: Parallel Tool Execution with Per-Tool Timeouts + Partial Results
=============================================================================
Production use-case: Agent needs to call 4 external APIs (SAP, CRM, ERP, weather)
simultaneously. Some are slow or unreliable. We fan out in parallel, enforce
per-tool timeouts, collect whatever succeeds, and generate an answer from
partial results rather than failing the whole request.

Key concepts:
- asyncio.gather with per-coroutine timeout using asyncio.wait_for
- LangGraph Send API for true parallel fan-out
- Partial result aggregation node
- Tool result typing with status (success | timeout | error)
- Synthesis prompt adapted to available vs missing results
"""

import asyncio
import random
import time
from typing import Annotated, Any
from uuid import uuid4

from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
import operator


# ── Models ────────────────────────────────────────────────────────────────────

class ToolResult(BaseModel):
    tool_name: str
    status: str       # success | timeout | error
    data: Any = None
    error_msg: str = ""
    duration_ms: float = 0.0


class ParallelToolState(BaseModel):
    query: str = ""
    tool_results: Annotated[list[ToolResult], operator.add] = []
    final_answer: str = ""
    session_id: str = Field(default_factory=lambda: str(uuid4()))


class SingleToolState(BaseModel):
    """State for individual tool worker node (used with Send API)."""
    query: str = ""
    tool_name: str = ""
    timeout_seconds: float = 5.0


# ── Tool implementations (simulated) ─────────────────────────────────────────

async def _call_sap_inventory(query: str) -> dict:
    """Calls SAP inventory API — simulate ~2s response."""
    await asyncio.sleep(2.0 + random.uniform(0, 0.5))
    return {"inventory_items": 1247, "low_stock_alerts": 3, "pending_orders": 12}


async def _call_crm_accounts(query: str) -> dict:
    """Calls CRM for supplier/customer account data — simulate ~1s response."""
    await asyncio.sleep(1.0 + random.uniform(0, 0.3))
    return {"active_suppliers": 89, "contracts_expiring_30d": 4, "open_tickets": 7}


async def _call_erp_financials(query: str) -> dict:
    """Calls ERP for financial data — simulate ~4s (sometimes times out)."""
    await asyncio.sleep(4.0 + random.uniform(0, 2.0))  # will often timeout at 3s
    return {"q3_spend_eur": 14_200_000, "budget_utilization_pct": 87.3}


async def _call_market_feed(query: str) -> dict:
    """Calls market data feed — simulate ~0.5s."""
    await asyncio.sleep(0.5)
    return {"crude_oil_eur_per_barrel": 74.2, "euro_usd": 1.082, "chemical_index": 102.3}


TOOL_REGISTRY: dict[str, tuple] = {
    # name → (coroutine_factory, timeout_seconds)
    "sap_inventory":  (_call_sap_inventory, 3.0),
    "crm_accounts":   (_call_crm_accounts, 3.0),
    "erp_financials": (_call_erp_financials, 3.0),   # intentionally tight
    "market_feed":    (_call_market_feed, 2.0),
}


# ── Nodes ─────────────────────────────────────────────────────────────────────

async def plan_tools(state: ParallelToolState) -> dict:
    """Decide which tools to call. In production: LLM-driven tool selection."""
    print(f"  [Planner] Will call {list(TOOL_REGISTRY.keys())} in parallel")
    return {}


def fan_out_tools(state: ParallelToolState) -> list[Send]:
    """
    LangGraph Send API: launches one worker node per tool simultaneously.
    Each Send creates an independent subgraph invocation.
    """
    return [
        Send("tool_worker", SingleToolState(
            query=state.query,
            tool_name=tool_name,
            timeout_seconds=timeout,
        ))
        for tool_name, (_, timeout) in TOOL_REGISTRY.items()
    ]


async def tool_worker(state: SingleToolState) -> dict:
    """
    Executes a single tool with a hard timeout.
    Returns a ToolResult regardless of outcome — never raises.
    """
    tool_fn, _ = TOOL_REGISTRY[state.tool_name]
    t0 = time.monotonic()

    try:
        data = await asyncio.wait_for(
            tool_fn(state.query),
            timeout=state.timeout_seconds,
        )
        duration = (time.monotonic() - t0) * 1000
        print(f"  [ToolWorker:{state.tool_name}] ✅ {duration:.0f}ms")
        result = ToolResult(
            tool_name=state.tool_name,
            status="success",
            data=data,
            duration_ms=duration,
        )
    except asyncio.TimeoutError:
        duration = (time.monotonic() - t0) * 1000
        print(f"  [ToolWorker:{state.tool_name}] ⏰ TIMEOUT after {duration:.0f}ms")
        result = ToolResult(
            tool_name=state.tool_name,
            status="timeout",
            error_msg=f"Timed out after {state.timeout_seconds}s",
            duration_ms=duration,
        )
    except Exception as e:
        duration = (time.monotonic() - t0) * 1000
        print(f"  [ToolWorker:{state.tool_name}] ❌ ERROR: {e}")
        result = ToolResult(
            tool_name=state.tool_name,
            status="error",
            error_msg=str(e),
            duration_ms=duration,
        )

    # Annotated[list, operator.add] merges results from all parallel workers
    return {"tool_results": [result]}


async def synthesize(state: ParallelToolState) -> dict:
    """
    Generates a final answer from whatever tool results succeeded.
    Adapts the prompt to explicitly acknowledge missing data.
    """
    successes = [r for r in state.tool_results if r.status == "success"]
    failures = [r for r in state.tool_results if r.status != "success"]

    context_parts = []
    for r in successes:
        context_parts.append(f"[{r.tool_name}]: {r.data}")

    missing_note = ""
    if failures:
        failed_names = [f.tool_name for f in failures]
        missing_note = f"\n\nNote: The following data sources were unavailable: {', '.join(failed_names)}. Your answer should acknowledge this."

    llm = AzureChatOpenAI(
        azure_deployment="gpt-4o",
        api_version="2024-10-21",
        temperature=0,
    )
    response = await llm.ainvoke([
        SystemMessage(content=f"""Answer the user's question using the available data below.
If some data sources are missing, acknowledge it and answer from what's available.

Available data:
{chr(10).join(context_parts)}
{missing_note}"""),
        HumanMessage(content=state.query),
    ])

    print(f"\n  [Synthesis] {len(successes)}/{len(state.tool_results)} tools succeeded")
    return {"final_answer": response.content}


# ── Graph ─────────────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(ParallelToolState)
    g.add_node("plan_tools", plan_tools)
    g.add_node("tool_worker", tool_worker)
    g.add_node("synthesize", synthesize)

    g.add_edge(START, "plan_tools")
    g.add_conditional_edges("plan_tools", fan_out_tools, ["tool_worker"])
    g.add_edge("tool_worker", "synthesize")
    g.add_edge("synthesize", END)

    return g.compile()


async def demo():
    print("=== Pattern 06: Parallel Tools + Timeouts ===\n")
    t0 = time.monotonic()
    graph = build_graph()

    result = await graph.ainvoke(ParallelToolState(
        query="Give me an operational summary: inventory status, financial position, and market conditions."
    ))

    wall_time = time.monotonic() - t0
    print(f"\n  Wall time: {wall_time:.1f}s (parallel, not sequential)")
    print(f"  Answer: {result.final_answer[:300]}...")


if __name__ == "__main__":
    asyncio.run(demo())

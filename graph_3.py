"""
Pattern 02: Multi-Agent Supervisor with Error Recovery + Dead Letter Queue
==========================================================================
Production use-case: A supervisor routes tasks to specialist sub-agents.
When a sub-agent fails (bad output, timeout, exception), the supervisor
retries with a fallback agent, downgrades gracefully, and if all retries
are exhausted, routes to a dead letter queue with full diagnostic context.

Key concepts:
- Supervisor node with structured routing
- Per-agent retry budget tracked in state
- Fallback agent chain (primary → fallback → last-resort)
- Dead letter queue node with diagnostic payload
- All errors captured without crashing the graph
"""

import asyncio
from typing import Annotated, Any, Literal
from dataclasses import dataclass, field

from langgraph.graph import END, START, StateGraph
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel
import operator


# ── State ─────────────────────────────────────────────────────────────────────

class AgentError(BaseModel):
    agent: str
    attempt: int
    error_type: str
    error_msg: str
    timestamp: str


class SupervisorState(BaseModel):
    task: str = ""
    route: str = ""                         # which agent to call next
    agent_outputs: dict[str, Any] = {}
    retry_counts: dict[str, int] = {}       # agent_name → attempts used
    errors: list[AgentError] = []
    final_answer: str = ""
    dlq_payload: dict[str, Any] | None = None   # populated on terminal failure
    status: str = "in_progress"             # in_progress | success | dead_letter


MAX_RETRIES_PER_AGENT = 2
AGENT_FALLBACK_CHAIN = {
    "data_analyst":  ["data_analyst", "data_analyst_fallback", "last_resort"],
    "doc_retriever": ["doc_retriever", "doc_retriever_fallback", "last_resort"],
    "summarizer":    ["summarizer",   "last_resort"],
}


# ── LLM helpers ──────────────────────────────────────────────────────────────

def get_llm(temperature: float = 0) -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_deployment="gpt-4o",
        api_version="2024-10-21",
        temperature=temperature,
    )


def get_cheap_llm() -> AzureChatOpenAI:
    """Fallback to a cheaper/faster model when primary is unreliable."""
    return AzureChatOpenAI(
        azure_deployment="gpt-4o-mini",
        api_version="2024-10-21",
        temperature=0,
    )


# ── Supervisor ────────────────────────────────────────────────────────────────

async def supervisor(state: SupervisorState) -> dict:
    """
    Routes the task to the right specialist. On retry, escalates through
    the fallback chain until MAX_RETRIES_PER_AGENT is exhausted.
    """
    # If already routed and failed, escalate that agent's fallback
    if state.route and state.errors:
        last_error = state.errors[-1]
        chain = AGENT_FALLBACK_CHAIN.get(state.route, ["last_resort"])
        attempts = state.retry_counts.get(state.route, 0)
        if attempts < len(chain):
            next_agent = chain[attempts]
            print(f"  [Supervisor] Escalating to fallback: {next_agent}")
            return {"route": next_agent}
        else:
            return {"route": "dead_letter", "status": "dead_letter"}

    # First-time routing via LLM
    llm = get_llm()
    response = await llm.ainvoke([
        SystemMessage(content="""You are a task router. Given a task, reply with ONLY one of:
data_analyst, doc_retriever, summarizer"""),
        HumanMessage(content=state.task)
    ])
    route = response.content.strip().lower()
    if route not in AGENT_FALLBACK_CHAIN:
        route = "summarizer"
    print(f"  [Supervisor] Routing to: {route}")
    return {"route": route}


# ── Sub-agents ────────────────────────────────────────────────────────────────

async def _run_agent(
    state: SupervisorState,
    agent_name: str,
    system_prompt: str,
    llm,
    simulate_failure_on_attempt: int | None = None,
) -> dict:
    from datetime import datetime

    attempts = state.retry_counts.get(state.route, 0)
    new_counts = {**state.retry_counts, state.route: attempts + 1}

    # Simulate flaky agent for demo purposes
    if simulate_failure_on_attempt is not None and attempts < simulate_failure_on_attempt:
        error = AgentError(
            agent=agent_name,
            attempt=attempts + 1,
            error_type="SimulatedFailure",
            error_msg=f"{agent_name} failed on attempt {attempts + 1} (flaky upstream)",
            timestamp=datetime.utcnow().isoformat(),
        )
        print(f"  [{agent_name}] ❌ Attempt {attempts + 1} failed")
        return {"retry_counts": new_counts, "errors": state.errors + [error]}

    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=state.task),
    ])
    output = response.content
    print(f"  [{agent_name}] ✅ Success on attempt {attempts + 1}")
    return {
        "retry_counts": new_counts,
        "agent_outputs": {**state.agent_outputs, agent_name: output},
        "final_answer": output,
        "status": "success",
    }


async def data_analyst(state: SupervisorState) -> dict:
    return await _run_agent(
        state, "data_analyst",
        "You are a precise data analyst. Analyze data questions concisely.",
        get_llm(),
        simulate_failure_on_attempt=1,  # fails first attempt to demo retry
    )


async def data_analyst_fallback(state: SupervisorState) -> dict:
    return await _run_agent(
        state, "data_analyst_fallback",
        "You are a backup data analyst. Answer with what you know, even if approximate.",
        get_cheap_llm(),
    )


async def doc_retriever(state: SupervisorState) -> dict:
    return await _run_agent(
        state, "doc_retriever",
        "You are a document retrieval agent. Return relevant facts from your knowledge.",
        get_llm(),
    )


async def doc_retriever_fallback(state: SupervisorState) -> dict:
    return await _run_agent(
        state, "doc_retriever_fallback",
        "You are a fallback retriever. Return the best possible answer without document access.",
        get_cheap_llm(),
    )


async def summarizer(state: SupervisorState) -> dict:
    return await _run_agent(
        state, "summarizer",
        "You are a summarizer. Produce a clear, concise summary.",
        get_llm(),
    )


async def last_resort(state: SupervisorState) -> dict:
    """Minimal effort response — better than nothing."""
    return await _run_agent(
        state, "last_resort",
        "Provide the most basic helpful response you can, even with very limited context.",
        get_cheap_llm(),
    )


async def dead_letter_queue(state: SupervisorState) -> dict:
    """
    All retries exhausted. Record full diagnostic payload.
    In production: write to a monitoring table, fire a PagerDuty alert,
    or push to an Azure Service Bus dead letter topic.
    """
    from datetime import datetime
    payload = {
        "task": state.task,
        "original_route": state.route,
        "total_attempts": sum(state.retry_counts.values()),
        "errors": [e.model_dump() for e in state.errors],
        "agent_outputs_partial": state.agent_outputs,
        "failed_at": datetime.utcnow().isoformat(),
        "recommended_action": "manual_review",
    }
    print(f"\n  [DLQ] ☠️  Task sent to dead letter queue after {payload['total_attempts']} total attempts")
    return {"dlq_payload": payload, "status": "dead_letter"}


# ── Routing logic ─────────────────────────────────────────────────────────────

def route_from_supervisor(state: SupervisorState) -> str:
    return state.route  # supervisor sets this directly


def should_continue(state: SupervisorState) -> str:
    if state.status in ("success", "dead_letter"):
        return END
    return "supervisor"   # loop back for retry routing


# ── Graph ─────────────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(SupervisorState)

    g.add_node("supervisor", supervisor)
    g.add_node("data_analyst", data_analyst)
    g.add_node("data_analyst_fallback", data_analyst_fallback)
    g.add_node("doc_retriever", doc_retriever)
    g.add_node("doc_retriever_fallback", doc_retriever_fallback)
    g.add_node("summarizer", summarizer)
    g.add_node("last_resort", last_resort)
    g.add_node("dead_letter_queue", dead_letter_queue)

    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", route_from_supervisor, {
        "data_analyst": "data_analyst",
        "data_analyst_fallback": "data_analyst_fallback",
        "doc_retriever": "doc_retriever",
        "doc_retriever_fallback": "doc_retriever_fallback",
        "summarizer": "summarizer",
        "last_resort": "last_resort",
        "dead_letter": "dead_letter_queue",
    })

    for agent in ["data_analyst", "data_analyst_fallback", "doc_retriever",
                  "doc_retriever_fallback", "summarizer", "last_resort"]:
        g.add_conditional_edges(agent, should_continue, {"supervisor": "supervisor", END: END})

    g.add_edge("dead_letter_queue", END)

    return g.compile()


async def demo():
    print("=== Pattern 02: Supervisor Error Recovery + DLQ ===\n")
    graph = build_graph()

    result = await graph.ainvoke(SupervisorState(
        task="Analyze Q3 procurement spend variance vs budget for the EMEA region."
    ))
    print(f"\n  Status: {result.status}")
    if result.status == "success":
        print(f"  Answer: {result.final_answer[:200]}...")
    else:
        print(f"  DLQ payload: {result.dlq_payload}")


if __name__ == "__main__":
    asyncio.run(demo())

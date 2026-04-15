"""
Pattern 07: Dynamic Routing with Confidence Scoring + Audit Trail
=================================================================
Production use-case: An enterprise triage agent receives mixed incoming
requests (procurement, HR, IT, legal, escalation). A router LLM classifies
each request with a confidence score. High-confidence routes go direct;
low-confidence routes go to a clarification node; very low confidence
escalates to a human. Every routing decision is written to an immutable
audit trail.

Key concepts:
- Structured router output (route + confidence + reasoning)
- Confidence thresholds: high (>0.85) → direct, medium (0.5–0.85) → clarify, low (<0.5) → escalate
- Clarification loop: re-route after user clarifies (max 2 rounds)
- Immutable audit trail (append-only list in state)
- Each routing decision logged with timestamp, confidence, reasoning
"""

import asyncio
import json
import re
from datetime import datetime, timezone
from typing import Annotated, Literal
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field
import operator


# ── Thresholds ────────────────────────────────────────────────────────────────

HIGH_CONFIDENCE = 0.85
LOW_CONFIDENCE = 0.50
MAX_CLARIFICATION_ROUNDS = 2

VALID_ROUTES = ["procurement", "hr", "it_support", "legal", "general", "escalate"]


# ── Models ────────────────────────────────────────────────────────────────────

class RouterDecision(BaseModel):
    route: str
    confidence: float          # 0.0–1.0
    reasoning: str
    clarification_question: str | None = None   # asked if confidence is medium


class AuditEntry(BaseModel):
    event_type: str            # route_decision | clarification_sent | escalation | resolution
    timestamp: str
    route: str
    confidence: float
    reasoning: str
    round: int
    message_snippet: str


class RouterState(BaseModel):
    messages: Annotated[list[BaseMessage], operator.add] = []
    current_route: str = ""
    confidence: float = 0.0
    clarification_rounds: int = 0
    audit_trail: Annotated[list[AuditEntry], operator.add] = []
    final_response: str = ""
    status: str = "routing"   # routing | clarifying | resolved | escalated


# ── LLM ───────────────────────────────────────────────────────────────────────

def get_llm() -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_deployment="gpt-4o",
        api_version="2024-10-21",
        temperature=0,
    )


def _log(entry: AuditEntry):
    print(f"  [Audit] {entry.event_type} → {entry.route} (conf={entry.confidence:.2f}) round={entry.round}")


# ── Router ────────────────────────────────────────────────────────────────────

async def route_request(state: RouterState) -> dict:
    """
    Classifies the incoming request with structured output.
    Uses the full conversation history so clarifications improve routing.
    """
    llm = get_llm()
    system = SystemMessage(content=f"""You are an enterprise request router.
Classify the request into exactly one of: {VALID_ROUTES}

Respond ONLY with valid JSON:
{{
  "route": "<one of the valid routes>",
  "confidence": <0.0-1.0>,
  "reasoning": "<one sentence>",
  "clarification_question": "<question to ask if confidence < {HIGH_CONFIDENCE}, else null>"
}}""")

    response = await llm.ainvoke([system] + state.messages)

    match = re.search(r'\{.*\}', response.content, re.DOTALL)
    try:
        data = json.loads(match.group()) if match else {}
        decision = RouterDecision(
            route=data.get("route", "general"),
            confidence=float(data.get("confidence", 0.5)),
            reasoning=data.get("reasoning", ""),
            clarification_question=data.get("clarification_question"),
        )
    except Exception:
        decision = RouterDecision(route="general", confidence=0.4, reasoning="parse error")

    audit = AuditEntry(
        event_type="route_decision",
        timestamp=datetime.now(timezone.utc).isoformat(),
        route=decision.route,
        confidence=decision.confidence,
        reasoning=decision.reasoning,
        round=state.clarification_rounds,
        message_snippet=str(state.messages[-1].content)[:100],
    )
    _log(audit)

    return {
        "current_route": decision.route,
        "confidence": decision.confidence,
        "audit_trail": [audit],
    }


async def ask_clarification(state: RouterState) -> dict:
    """Ask the user to clarify their request. Max MAX_CLARIFICATION_ROUNDS times."""
    llm = get_llm()
    response = await llm.ainvoke([
        SystemMessage(content="You are a helpful assistant. Ask one short, specific clarification question to better understand the user's request."),
    ] + state.messages)

    question = response.content
    audit = AuditEntry(
        event_type="clarification_sent",
        timestamp=datetime.now(timezone.utc).isoformat(),
        route=state.current_route,
        confidence=state.confidence,
        reasoning=f"Confidence {state.confidence:.2f} below threshold, requesting clarification",
        round=state.clarification_rounds + 1,
        message_snippet=question[:100],
    )
    _log(audit)

    # In production: send this question back to the UI/channel and interrupt
    # For demo: simulate user answering
    simulated_answer = f"[User clarification round {state.clarification_rounds + 1}]: I need to update the vendor payment terms in our procurement system."
    print(f"  [Clarification] Q: {question[:80]}")
    print(f"  [Clarification] A (simulated): {simulated_answer[:80]}")

    return {
        "messages": [AIMessage(content=question), HumanMessage(content=simulated_answer)],
        "clarification_rounds": state.clarification_rounds + 1,
        "audit_trail": [audit],
        "status": "routing",
    }


# ── Specialist handlers ───────────────────────────────────────────────────────

async def _specialist_handler(state: RouterState, department: str, persona: str) -> dict:
    llm = get_llm()
    response = await llm.ainvoke([
        SystemMessage(content=f"You are the {department} department assistant. {persona}"),
    ] + state.messages)

    audit = AuditEntry(
        event_type="resolution",
        timestamp=datetime.now(timezone.utc).isoformat(),
        route=department,
        confidence=state.confidence,
        reasoning=f"Handled by {department}",
        round=state.clarification_rounds,
        message_snippet=response.content[:100],
    )
    _log(audit)
    print(f"  [{department.upper()}] Handled with conf={state.confidence:.2f}")
    return {"final_response": response.content, "status": "resolved", "audit_trail": [audit]}


async def handle_procurement(state: RouterState) -> dict:
    return await _specialist_handler(state, "procurement", "Help with purchasing, vendor management, and contracts.")

async def handle_hr(state: RouterState) -> dict:
    return await _specialist_handler(state, "hr", "Help with people matters, leave, benefits, and HR policies.")

async def handle_it(state: RouterState) -> dict:
    return await _specialist_handler(state, "it_support", "Help with technical issues, access, and software.")

async def handle_legal(state: RouterState) -> dict:
    return await _specialist_handler(state, "legal", "Help with compliance, contracts, and legal questions.")

async def handle_general(state: RouterState) -> dict:
    return await _specialist_handler(state, "general", "Provide a helpful general response.")


async def escalate_to_human(state: RouterState) -> dict:
    """Low confidence + max clarifications exhausted → escalate."""
    audit = AuditEntry(
        event_type="escalation",
        timestamp=datetime.now(timezone.utc).isoformat(),
        route="human_agent",
        confidence=state.confidence,
        reasoning=f"Confidence {state.confidence:.2f} after {state.clarification_rounds} rounds — escalating",
        round=state.clarification_rounds,
        message_snippet=str(state.messages[-1].content)[:100],
    )
    _log(audit)

    ticket_id = f"ESC-{uuid4().hex[:8].upper()}"
    response = (f"Your request has been escalated to a human agent. "
                f"Ticket ID: {ticket_id}. Expected response time: 2 business hours.")
    print(f"  [ESCALATE] → {ticket_id}")
    return {"final_response": response, "status": "escalated", "audit_trail": [audit]}


# ── Routing logic ─────────────────────────────────────────────────────────────

def decide_next(state: RouterState) -> str:
    if state.confidence >= HIGH_CONFIDENCE:
        return state.current_route

    if state.confidence < LOW_CONFIDENCE:
        if state.clarification_rounds >= MAX_CLARIFICATION_ROUNDS:
            return "escalate_to_human"
        return "ask_clarification"

    # Medium confidence (0.5–0.85)
    if state.clarification_rounds >= MAX_CLARIFICATION_ROUNDS:
        return state.current_route   # best-effort route
    return "ask_clarification"


def route_after_clarification(state: RouterState) -> str:
    return "route_request"   # always re-route after clarification


# ── Graph ─────────────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(RouterState)
    g.add_node("route_request", route_request)
    g.add_node("ask_clarification", ask_clarification)
    g.add_node("procurement", handle_procurement)
    g.add_node("hr", handle_hr)
    g.add_node("it_support", handle_it)
    g.add_node("legal", handle_legal)
    g.add_node("general", handle_general)
    g.add_node("escalate_to_human", escalate_to_human)

    g.add_edge(START, "route_request")
    g.add_conditional_edges("route_request", decide_next, {
        "procurement": "procurement",
        "hr": "hr",
        "it_support": "it_support",
        "legal": "legal",
        "general": "general",
        "ask_clarification": "ask_clarification",
        "escalate_to_human": "escalate_to_human",
    })
    g.add_edge("ask_clarification", "route_request")
    for dept in ["procurement", "hr", "it_support", "legal", "general", "escalate_to_human"]:
        g.add_edge(dept, END)

    return g.compile()


async def demo():
    print("=== Pattern 07: Dynamic Routing + Confidence + Audit Trail ===\n")
    graph = build_graph()

    # High confidence — routes directly
    print("--- Case 1: Clear procurement request ---")
    r1 = await graph.ainvoke(RouterState(
        messages=[HumanMessage(content="I need to raise a purchase order for 200 units of industrial solvent from BASF.")]
    ))
    print(f"  Route: {r1.current_route}, Status: {r1.status}")
    print(f"  Audit entries: {len(r1.audit_trail)}\n")

    # Ambiguous — triggers clarification
    print("--- Case 2: Ambiguous request → clarification → re-route ---")
    r2 = await graph.ainvoke(RouterState(
        messages=[HumanMessage(content="I need to update some terms in the system.")]
    ))
    print(f"  Route: {r2.current_route}, Status: {r2.status}")
    print(f"  Clarification rounds: {r2.clarification_rounds}")
    print(f"  Full audit trail ({len(r2.audit_trail)} entries):")
    for entry in r2.audit_trail:
        print(f"    [{entry.event_type}] {entry.route} conf={entry.confidence:.2f}")


if __name__ == "__main__":
    asyncio.run(demo())

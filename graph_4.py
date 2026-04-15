"""
Pattern 01: Interrupt / Resume with External Approval Callback
==============================================================
Production use-case: Agent proposes a purchase order or supplier action,
execution pauses, an external system (email, SAP workflow, Slack) delivers
human approval, then the graph resumes from the exact checkpoint.

Key concepts:
- NodeInterrupt to pause mid-graph
- SqliteSaver checkpointer (swap for AsyncPostgresSaver in production)
- Resuming with Command(resume=...) after external callback
- Thread-scoped state so concurrent requests don't collide
"""

import asyncio
import uuid
from datetime import datetime
from typing import Annotated, Any

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel
import operator


# ── State ────────────────────────────────────────────────────────────────────

class ApprovalState(BaseModel):
    messages: Annotated[list[BaseMessage], add_messages] = []
    proposed_action: dict[str, Any] | None = None
    approval_status: str = "pending"          # pending | approved | rejected
    approval_metadata: dict[str, Any] = {}
    execution_result: dict[str, Any] | None = None
    thread_id: str = ""
    created_at: str = ""


# ── LLM ─────────────────────────────────────────────────────────────────────

def get_llm() -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_deployment="gpt-4o",
        api_version="2024-10-21",
        temperature=0,
        max_tokens=2048,
        # azure_endpoint and api_key from env: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY
    )


# ── Nodes ────────────────────────────────────────────────────────────────────

async def analyze_request(state: ApprovalState) -> dict:
    """LLM analyzes the incoming request and proposes a concrete action."""
    llm = get_llm()
    system = SystemMessage(content="""You are a procurement assistant.
Analyze the user's request and produce a structured proposed action.
Always respond with JSON in this exact shape:
{
  "action_type": "purchase_order | supplier_contact | contract_amendment",
  "supplier": "<name>",
  "amount_eur": <float>,
  "justification": "<one sentence>",
  "urgency": "low | medium | high",
  "requires_approval_level": "team_lead | director | cfo"
}""")

    response = await llm.ainvoke([system] + state.messages)
    import json, re
    match = re.search(r'\{.*\}', response.content, re.DOTALL)
    proposed = json.loads(match.group()) if match else {"raw": response.content}

    return {
        "messages": [response],
        "proposed_action": proposed,
    }


async def request_human_approval(state: ApprovalState) -> dict:
    """
    Interrupt the graph here. The caller receives the checkpoint ID,
    sends it to an external approval system (email, SAP, Slack bot),
    and later calls resume_graph() with the approval decision.
    """
    approval_payload = interrupt({
        "message": "Action requires human approval before execution.",
        "proposed_action": state.proposed_action,
        "approval_level_required": state.proposed_action.get("requires_approval_level"),
        "checkpoint_hint": "Call resume_graph(thread_id, approved=True/False, approver=...) to continue.",
    })

    # approval_payload is whatever was passed to Command(resume=...) by the caller
    approved: bool = approval_payload.get("approved", False)
    approver: str = approval_payload.get("approver", "unknown")
    note: str = approval_payload.get("note", "")

    return {
        "approval_status": "approved" if approved else "rejected",
        "approval_metadata": {
            "approver": approver,
            "note": note,
            "decided_at": datetime.utcnow().isoformat(),
        },
    }


async def execute_action(state: ApprovalState) -> dict:
    """Execute the approved action — in production this calls SAP SOAP / REST APIs."""
    action = state.proposed_action
    # Simulate execution; replace with actual SAP PI/PO SOAP call
    result = {
        "status": "executed",
        "po_number": f"PO-{uuid.uuid4().hex[:8].upper()}",
        "supplier": action.get("supplier"),
        "amount_eur": action.get("amount_eur"),
        "executed_at": datetime.utcnow().isoformat(),
    }
    summary = AIMessage(content=f"✅ Purchase order {result['po_number']} created for {result['supplier']} (€{result['amount_eur']:,.2f}).")
    return {"execution_result": result, "messages": [summary]}


async def reject_action(state: ApprovalState) -> dict:
    note = state.approval_metadata.get("note", "No reason provided.")
    msg = AIMessage(content=f"❌ Action rejected by {state.approval_metadata.get('approver')}. Note: {note}")
    return {"messages": [msg]}


# ── Routing ───────────────────────────────────────────────────────────────────

def route_after_approval(state: ApprovalState) -> str:
    return "execute_action" if state.approval_status == "approved" else "reject_action"


# ── Graph ─────────────────────────────────────────────────────────────────────

def build_graph(checkpointer):
    g = StateGraph(ApprovalState)
    g.add_node("analyze_request", analyze_request)
    g.add_node("request_human_approval", request_human_approval)
    g.add_node("execute_action", execute_action)
    g.add_node("reject_action", reject_action)

    g.add_edge(START, "analyze_request")
    g.add_edge("analyze_request", "request_human_approval")
    g.add_conditional_edges("request_human_approval", route_after_approval)
    g.add_edge("execute_action", END)
    g.add_edge("reject_action", END)

    return g.compile(checkpointer=checkpointer)


# ── Public API ────────────────────────────────────────────────────────────────

async def start_request(user_message: str, thread_id: str | None = None) -> dict:
    """
    Start a new approval workflow. Returns thread_id + interrupt payload
    so your external system knows what to approve.
    """
    thread_id = thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
        graph = build_graph(checkpointer)
        initial_state = ApprovalState(
            messages=[HumanMessage(content=user_message)],
            thread_id=thread_id,
            created_at=datetime.utcnow().isoformat(),
        )
        result = await graph.ainvoke(initial_state, config=config)

    return {"thread_id": thread_id, "state": result}


async def resume_graph(thread_id: str, approved: bool, approver: str, note: str = "") -> dict:
    """
    Called by your external approval webhook (email reply parser, SAP callback, Slack bot).
    Resumes the graph from the interrupt checkpoint.
    """
    config = {"configurable": {"thread_id": thread_id}}
    resume_payload = {"approved": approved, "approver": approver, "note": note}

    async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
        graph = build_graph(checkpointer)
        result = await graph.ainvoke(Command(resume=resume_payload), config=config)

    return {"thread_id": thread_id, "final_state": result}


# ── Demo ──────────────────────────────────────────────────────────────────────

async def demo():
    print("=== Pattern 01: Interrupt / Resume Approval ===\n")

    # Step 1: Agent proposes action, graph pauses at interrupt
    print("Step 1: Submitting procurement request...")
    result = await start_request(
        "We need to emergency-order 500kg of specialty solvent from BASF. Budget ~€45,000.",
        thread_id="demo-thread-001"
    )
    print(f"  Thread ID: {result['thread_id']}")
    print(f"  Status: graph paused, awaiting external approval\n")

    # Step 2: Simulate external approver (e.g. SAP workflow callback after 30s)
    print("Step 2: Director approves via SAP workflow callback...")
    final = await resume_graph(
        thread_id="demo-thread-001",
        approved=True,
        approver="Dr. Klaus Bergmann (Director Procurement)",
        note="Confirmed urgent need. Proceed."
    )
    print(f"  Final messages:")
    for msg in final["final_state"].messages[-2:]:
        print(f"    [{msg.__class__.__name__}]: {msg.content}")


if __name__ == "__main__":
    asyncio.run(demo())

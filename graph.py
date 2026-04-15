"""
Pattern 08: Stateful Multi-Turn Memory with Context Window Management
=====================================================================
Production use-case: A long-running enterprise assistant that serves
users across multiple sessions. Messages accumulate and eventually
overflow the context window. This pattern uses a sliding window +
background summarization to maintain coherent long conversations
without token explosion.

Key concepts:
- PostgreSQL-backed checkpointing (persistent across restarts)
- Sliding window: keep last N messages in hot context
- Background summarization: older messages → compressed summary
- Summary injected as a special system message
- Memory tiers: working (last 10) | episodic (summary) | semantic (RAG)
- Session isolation via thread_id
"""

import asyncio
import json
from datetime import datetime, timezone
from typing import Annotated
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import (
    AIMessage, BaseMessage, HumanMessage, SystemMessage, RemoveMessage
)
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


# ── Configuration ─────────────────────────────────────────────────────────────

WORKING_MEMORY_SIZE = 10       # keep last N messages in full
SUMMARIZATION_TRIGGER = 16     # summarize when messages exceed this
MAX_SUMMARY_AGE_TURNS = 5      # re-summarize after N turns since last summary


# ── State ─────────────────────────────────────────────────────────────────────

class MemoryState(BaseModel):
    messages: Annotated[list[BaseMessage], add_messages] = []
    episodic_summary: str = ""            # compressed older context
    summary_turn_count: int = 0           # turns since last summarization
    user_profile: dict = {}               # persistent user facts (name, role, preferences)
    session_id: str = ""
    turn_count: int = 0


# ── LLM ───────────────────────────────────────────────────────────────────────

def get_llm() -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_deployment="gpt-4o",
        api_version="2024-10-21",
        temperature=0,
    )


# ── Nodes ─────────────────────────────────────────────────────────────────────

async def check_memory(state: MemoryState) -> dict:
    """
    Decides if we need to compress older messages into the episodic summary.
    Runs before each assistant turn.
    """
    msg_count = len(state.messages)
    turns_since_summary = state.summary_turn_count

    needs_summarization = (
        msg_count > SUMMARIZATION_TRIGGER or
        turns_since_summary >= MAX_SUMMARY_AGE_TURNS
    )

    if needs_summarization and msg_count > WORKING_MEMORY_SIZE:
        print(f"  [Memory] Summarizing: {msg_count} messages, {turns_since_summary} turns since last summary")
        return {"_needs_summarization": True}

    print(f"  [Memory] OK: {msg_count} messages, {turns_since_summary} turns since last summary")
    return {}


async def summarize_history(state: MemoryState) -> dict:
    """
    Compresses messages older than the working window into episodic summary.
    Merges with existing summary. Removes compressed messages from state.
    """
    messages = state.messages
    to_summarize = messages[:-WORKING_MEMORY_SIZE]
    to_keep = messages[-WORKING_MEMORY_SIZE:]

    if not to_summarize:
        return {}

    llm = get_llm()
    existing = f"\nExisting summary:\n{state.episodic_summary}" if state.episodic_summary else ""
    conversation_text = "\n".join(
        f"{msg.__class__.__name__}: {msg.content}" for msg in to_summarize
    )

    response = await llm.ainvoke([
        SystemMessage(content=f"""Compress this conversation history into a dense summary.
Preserve: key decisions made, user preferences revealed, important facts, open questions.
Be specific — include names, numbers, and concrete outcomes.{existing}"""),
        HumanMessage(content=conversation_text),
    ])

    new_summary = response.content
    print(f"  [Summarize] Compressed {len(to_summarize)} messages → {len(new_summary)} chars")

    # Remove old messages from state using RemoveMessage
    removals = [RemoveMessage(id=msg.id) for msg in to_summarize if hasattr(msg, 'id') and msg.id]

    return {
        "messages": removals,
        "episodic_summary": new_summary,
        "summary_turn_count": 0,
    }


async def extract_user_facts(state: MemoryState) -> dict:
    """
    Lightweight pass to extract persistent user facts from the latest message.
    Updates user profile without re-processing full history.
    """
    if not state.messages:
        return {}

    last_user_msg = next(
        (m for m in reversed(state.messages) if isinstance(m, HumanMessage)), None
    )
    if not last_user_msg:
        return {}

    llm = get_llm()
    response = await llm.ainvoke([
        SystemMessage(content=f"""Extract any new persistent facts about the user from this message.
Only extract clearly stated facts (name, role, company, preferences, constraints).
Current profile: {json.dumps(state.user_profile)}
Respond ONLY with JSON delta (new/changed keys only), or {{}} if nothing new."""),
        HumanMessage(content=str(last_user_msg.content)),
    ])

    try:
        import re
        match = re.search(r'\{.*\}', response.content, re.DOTALL)
        delta = json.loads(match.group()) if match else {}
        if delta:
            updated_profile = {**state.user_profile, **delta}
            print(f"  [UserFacts] Updated profile: {delta}")
            return {"user_profile": updated_profile}
    except Exception:
        pass

    return {}


async def generate_response(state: MemoryState) -> dict:
    """
    Main generation node. Builds context from three tiers:
    1. User profile (persistent facts)
    2. Episodic summary (compressed history)
    3. Working memory (recent messages)
    """
    llm = get_llm()

    # Build system message with all memory tiers
    memory_context_parts = []
    if state.user_profile:
        memory_context_parts.append(f"User profile: {json.dumps(state.user_profile)}")
    if state.episodic_summary:
        memory_context_parts.append(f"Earlier conversation summary:\n{state.episodic_summary}")

    memory_context = "\n\n".join(memory_context_parts)
    system_content = f"""You are a knowledgeable enterprise assistant with memory.
{f'Context from earlier in this conversation:{chr(10)}{memory_context}' if memory_context else ''}

Be concise, precise, and reference earlier context naturally when relevant."""

    messages_to_send = [SystemMessage(content=system_content)] + state.messages

    response = await llm.ainvoke(messages_to_send)
    print(f"  [Generate] Turn {state.turn_count + 1}, working_memory={len(state.messages)} msgs")

    return {
        "messages": [response],
        "turn_count": state.turn_count + 1,
        "summary_turn_count": state.summary_turn_count + 1,
    }


# ── Routing ───────────────────────────────────────────────────────────────────

def should_summarize(state: MemoryState) -> str:
    msg_count = len(state.messages)
    needs = msg_count > SUMMARIZATION_TRIGGER and state.summary_turn_count >= MAX_SUMMARY_AGE_TURNS
    return "summarize_history" if needs else "extract_user_facts"


# ── Graph ─────────────────────────────────────────────────────────────────────

def build_graph(checkpointer):
    g = StateGraph(MemoryState)
    g.add_node("check_memory", check_memory)
    g.add_node("summarize_history", summarize_history)
    g.add_node("extract_user_facts", extract_user_facts)
    g.add_node("generate_response", generate_response)

    g.add_edge(START, "check_memory")
    g.add_conditional_edges("check_memory", should_summarize, {
        "summarize_history": "summarize_history",
        "extract_user_facts": "extract_user_facts",
    })
    g.add_edge("summarize_history", "extract_user_facts")
    g.add_edge("extract_user_facts", "generate_response")
    g.add_edge("generate_response", END)

    return g.compile(checkpointer=checkpointer)


# ── Public API ────────────────────────────────────────────────────────────────

async def chat(
    user_message: str,
    thread_id: str,
    db_path: str = "memory.db",
) -> str:
    """
    Stateful chat function. Same thread_id → same conversation history.
    Different thread_id → isolated session.
    """
    config = {"configurable": {"thread_id": thread_id}}

    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
        graph = build_graph(checkpointer)
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=user_message)]},
            config=config,
        )

    last_ai = next(
        (m for m in reversed(result["messages"]) if isinstance(m, AIMessage)),
        None
    )
    return last_ai.content if last_ai else ""


# ── Demo ──────────────────────────────────────────────────────────────────────

async def demo():
    print("=== Pattern 08: Stateful Multi-Turn Memory ===\n")
    thread = f"demo-{uuid4().hex[:8]}"

    turns = [
        "Hi, I'm Priya, Head of Procurement at Novartis Basel. I manage a €50M annual spend.",
        "We're trying to consolidate our solvent suppliers from 12 to 5. What framework would you recommend?",
        "For the scoring criteria, we want to weight sustainability at 30%. Thoughts on the other weights?",
        "Actually, our CEO just said we need to hit 40% sustainability. Can we adjust the framework?",
        "What was the original framework you suggested before the sustainability change?",  # tests memory
    ]

    for i, msg in enumerate(turns, 1):
        print(f"\nTurn {i}: {msg[:60]}...")
        response = await chat(msg, thread_id=thread)
        print(f"Assistant: {response[:200]}...")


if __name__ == "__main__":
    asyncio.run(demo())

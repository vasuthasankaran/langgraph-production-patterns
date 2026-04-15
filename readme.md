# LangGraph Production Patterns

> Tested patterns for building **production-grade agentic AI systems** with LangGraph and Azure OpenAI.

These aren't tutorial snippets. Each pattern solves a real problem encountered shipping multi-agent pipelines at enterprise scale — covering reliability, observability, cost control, and human oversight.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.3+-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![Azure OpenAI](https://img.shields.io/badge/Azure%20OpenAI-GPT--4o-blue.svg)](https://azure.microsoft.com/en-us/products/ai-services/openai-service)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

---

## Patterns

| # | Pattern | The Problem It Solves |
|---|---------|----------------------|
| 01 | [Interrupt / Resume with External Approval](#01-interrupt--resume-with-external-approval) | Agent proposes an action; execution pauses for human approval via external system (SAP, email, Slack), then resumes from exact checkpoint |
| 02 | [Supervisor with Error Recovery + DLQ](#02-supervisor-with-error-recovery--dead-letter-queue) | Sub-agents fail silently or return bad output; supervisor retries through fallback chain, routes to dead letter queue with full diagnostics |
| 03 | [Streaming RAG with Citation Tracking](#03-streaming-rag-with-azure-ai-search--citation-tracking) | Enterprise doc Q&A where every answer claim must trace back to a source document, chunk, and page number |
| 04 | [LangFuse Production Observability](#04-langfuse-production-observability) | LLM costs, latencies, and quality scores are invisible in production; full trace coverage with LLM-as-judge eval |
| 05 | [Token Budget Guardian](#05-token-budget-guardian) | Long-running agents silently exceed token budgets; mid-graph enforcement with soft (compress) and hard (abort) limits |
| 06 | [Parallel Tools with Timeouts + Partial Results](#06-parallel-tools-with-per-tool-timeouts--partial-results) | Calling 4 APIs sequentially takes 10s and one timeout kills everything; fan-out parallel execution with partial result synthesis |
| 07 | [Dynamic Routing with Confidence + Audit Trail](#07-dynamic-routing-with-confidence-scoring--audit-trail) | Classifier routes requests with no uncertainty handling; confidence scoring, clarification loop, escalation, and immutable audit log |
| 08 | [Stateful Multi-Turn Memory](#08-stateful-multi-turn-memory-with-context-window-management) | Long conversations overflow context windows; sliding window + background summarization + persistent user profile across sessions |

---

## Quickstart

```bash
git clone https://github.com/YOUR_USERNAME/langgraph-production-patterns.git
cd langgraph-production-patterns

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Fill in AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY

# Run any pattern demo
python -m patterns.01_interrupt_resume.graph
python -m patterns.02_supervisor_recovery.graph
# ... etc
```

---

## Pattern Details

### 01 Interrupt / Resume with External Approval

**Problem:** Agent needs to propose a €45k purchase order, but execution must wait for a director to approve via SAP workflow. The graph must pause indefinitely, survive restarts, and resume from the exact state after approval arrives via webhook.

**Solution:** `interrupt()` pauses the graph at the approval node. State is persisted to SQLite (swap for PostgreSQL in prod). An external callback calls `Command(resume=payload)` to continue.

```python
# Start a workflow — graph pauses at approval interrupt
result = await start_request("Order 500kg solvent from BASF ~€45k", thread_id="req-001")

# Later — SAP workflow callback resumes the graph
final = await resume_graph(
    thread_id="req-001",
    approved=True,
    approver="Dr. Klaus Bergmann",
    note="Confirmed urgent."
)
```

**Key files:** [`patterns/01_interrupt_resume/graph.py`](./patterns/01_interrupt_resume/graph.py)

---

### 02 Supervisor with Error Recovery + Dead Letter Queue

**Problem:** A data analyst sub-agent returns garbage output on the first attempt. Without explicit retry logic, the supervisor either crashes or returns a bad answer silently.

**Solution:** Per-agent retry budgets in state. Supervisor re-routes through a fallback chain (`primary → fallback → last_resort`). After all retries exhausted, routes to a DLQ node that captures full diagnostics for monitoring.

```python
# Flaky agent fails on attempt 1 → supervisor escalates to fallback automatically
result = await graph.ainvoke(SupervisorState(
    task="Analyze Q3 procurement spend variance for EMEA."
))
# result.status == "success" (handled by fallback agent)
# result.errors contains the full failure trace
```

**Key files:** [`patterns/02_supervisor_recovery/graph.py`](./patterns/02_supervisor_recovery/graph.py)

---

### 03 Streaming RAG with Azure AI Search + Citation Tracking

**Problem:** Enterprise RAG systems answer questions but provide no traceability. Compliance and audit teams need to know which document, which page, and which chunk supported each claim.

**Solution:** Hybrid retrieval (vector + BM25 + semantic reranking). LLM instructed to annotate every factual sentence with `[CITE:chunk_id]`. Post-processing extracts a structured citation map. Confidence scoring prevents hallucination when retrieval is weak.

```python
result = await graph.ainvoke(RAGState(query="What is our supplier onboarding SLA?"))

# result.answer — clean answer without citation markers
# result.citations — [{source_document, page_number, excerpt, relevance_score}]
# result.confidence — 0.0–1.0 retrieval confidence score
```

**Key files:** [`patterns/03_streaming_rag/graph.py`](./patterns/03_streaming_rag/graph.py)

---

### 04 LangFuse Production Observability

**Problem:** LLM costs and latencies are invisible until the Azure bill arrives. Quality regressions after prompt changes go undetected. No audit trail for compliance.

**Solution:** LangFuse CallbackHandler on every LLM call. Manual spans for non-LLM steps (retrieval, tool calls). LLM-as-judge scores faithfulness and relevance, written back to LangFuse after every request. Per-session cost accumulation.

```python
# Every run produces a LangFuse trace with:
# - Full message tree (system, user, assistant)
# - Token counts + cost per call
# - Retrieval latency as a custom span
# - faithfulness + relevance eval scores
result = await graph.ainvoke(ObservableState(
    query="Q3 EMEA revenue performance?",
    user_id="user-123",
))
# result.total_cost_usd, result.eval_score, result.latencies_ms
```

**Key files:** [`patterns/04_langfuse/graph.py`](./patterns/04_langfuse/graph.py)

---

### 05 Token Budget Guardian

**Problem:** An agent loads 6 document chunks, runs 3 LLM calls, and burns through the context window silently. Either the call fails with a 413 or you get a surprise $0.80 charge per request.

**Solution:** Budget guardian node sits between every expensive step. Checks cumulative token spend against soft and hard limits. Soft limit → compress context with cheap model, continue. Hard limit → generate best-effort partial answer, stop.

```python
result = await graph.ainvoke(BudgetedState(
    messages=[HumanMessage(content="Summarize Q3 EMEA performance.")],
    budget=BudgetConfig(
        soft_limit_tokens=6_000,    # compress here
        hard_limit_tokens=10_000,   # abort here
        max_cost_usd=0.10,
    )
))
# result.compression_applied — True if soft limit was hit
# result.answer — may be prefixed with [PARTIAL] if hard limit hit
```

**Key files:** [`patterns/05_token_budget/graph.py`](./patterns/05_token_budget/graph.py)

---

### 06 Parallel Tools with Per-Tool Timeouts + Partial Results

**Problem:** Agent needs data from SAP, CRM, ERP, and market feed. Sequential calls take 8+ seconds. One slow ERP call (4s) times out and kills the entire request.

**Solution:** LangGraph `Send` API fans out all tool calls simultaneously. Each worker has an individual `asyncio.wait_for` timeout. Failed/timed-out tools produce a `ToolResult(status="timeout")`. Synthesis node adapts the prompt to explicitly acknowledge missing data sources.

```python
# 4 tools called in parallel — total wall time = slowest successful tool
result = await graph.ainvoke(ParallelToolState(
    query="Operational summary: inventory, financials, and market conditions."
))
# Wall time: ~2s (not 8s sequential)
# ERP times out → answer acknowledges missing financial data
```

**Key files:** [`patterns/06_parallel_tools/graph.py`](./patterns/06_parallel_tools/graph.py)

---

### 07 Dynamic Routing with Confidence Scoring + Audit Trail

**Problem:** A simple LLM classifier routes requests but has no uncertainty handling. Ambiguous requests get silently misrouted. No audit log for compliance.

**Solution:** Router outputs structured JSON with `confidence` score. High (>0.85) → direct route. Medium (0.5–0.85) → clarification loop (max 2 rounds). Low (<0.5) after clarification → escalate to human with ticket ID. Every routing decision appended to an immutable audit trail in state.

```python
result = await graph.ainvoke(RouterState(
    messages=[HumanMessage(content="I need to update some terms in the system.")]
))
# Low confidence → asks clarification → re-routes to procurement
# result.audit_trail contains every decision with timestamp + reasoning
```

**Key files:** [`patterns/07_dynamic_routing/graph.py`](./patterns/07_dynamic_routing/graph.py)

---

### 08 Stateful Multi-Turn Memory with Context Window Management

**Problem:** Long enterprise conversations accumulate hundreds of messages. After ~20 turns, the context window overflows. Naive truncation loses critical earlier context. Starting fresh each session means re-establishing facts every time.

**Solution:** Three-tier memory: working (last 10 messages, full fidelity) + episodic (compressed summary of older messages) + semantic (user profile facts extracted and persisted). Background summarization triggers after N turns. SQLite checkpointer persists across restarts.

```python
# Same thread_id = same conversation history across sessions
await chat("I'm Olive, Head of Procurement at Bayer.", thread_id="olive-001")
await chat("We want to consolidate suppliers from 12 to 5.", thread_id="olive-001")
# ... 10 more turns — automatic summarization triggers
await chat("What was the framework you suggested earlier?", thread_id="olive-001")
# Correctly references earlier turns via episodic summary
```

**Key files:** [`patterns/08_multiturn_memory/graph.py`](./patterns/08_multiturn_memory/graph.py)

---

## Stack

| Component | Technology |
|-----------|-----------|
| Agent framework | [LangGraph](https://langchain-ai.github.io/langgraph/) 0.3+ |
| LLM | [Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service) GPT-4o / GPT-4o-mini |
| Vector search | [Azure AI Search](https://azure.microsoft.com/en-us/products/ai-services/ai-search) (hybrid + semantic reranking) |
| Observability | [LangFuse](https://langfuse.com) |
| Checkpointing | SQLite (dev) → PostgreSQL (prod) |
| Token counting | [tiktoken](https://github.com/openai/tiktoken) |

---

## Production Notes

**Checkpointer:** Patterns 01 and 08 use `AsyncSqliteSaver` for simplicity. In production, replace with `AsyncPostgresSaver` from `langgraph-checkpoint-postgres`:

```python
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
async with AsyncPostgresSaver.from_conn_string(os.environ["DATABASE_URL"]) as checkpointer:
    graph = build_graph(checkpointer)
```

**LangGraph version:** These patterns target LangGraph 0.3+. If you see `ImportError` from `langgraph_prebuilt`, upgrade: `pip install langgraph>=0.3.0 langgraph-prebuilt>=0.1.6`.

**Azure deployments:** All patterns default to `gpt-4o` and `gpt-4o-mini`. Update deployment names in `shared/llm.py` to match your Azure resource.

---

## Author

**Vasuthalakshmi Sankaran**  
AI Solutions Architect — LangGraph · Azure OpenAI · Multi-Agent Systems  
Building production agentic pipelines for enterprise at scale.

[LinkedIn](https://www.linkedin.com/in/vasuthalakshmi-sankaran-1ab5995b/) · [GitHub](https://github.com/YOUR_USERNAME)

---

## License

MIT

# Application Architecture

A triage system where customers submit support queries that are routed, analyzed, and answered by a chain of AI agents. Business rules are enforced by Python code after the agents finish, then the result is returned to the client via an HTTP API, browser UI, or CLI.

---

## Component Map

```
 ENTRY POINTS
 ┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
 │       api.py         │  │        ui.py          │  │       main.py        │
 │  FastAPI  port 8000  │  │  NiceGUI  port 8080   │  │  CLI / local tests   │
 │  POST /triage        │  │  Browser form UI      │  │  asyncio.gather()    │
 │  POST /triage/stream │  │  Live status updates  │  │  Human-in-the-loop   │
 │  GET  /health        │  │  Real-time results    │  │  parallel requests   │
 └──────────┬───────────┘  └──────────┬────────────┘  └──────────┬───────────┘
            │                         │                           │
            └─────────────────────────┼───────────────────────────┘
                                      │  each creates per-request
                                      ▼
                         AppContext(db, user_email, on_status)

 SERVICE LAYER  ──  triage_service.py
 ┌─────────────────────────────────────────────────────────────────────────┐
 │  run_triage()                run_triage_streaming()                     │
 │  Standard run. Returns       Streaming run. Prints partial tokens       │
 │  complete FinalTriageResp.   to terminal. Returns FinalTriageResp.      │
 │                                                                         │
 │  run_triage_stream_events()                                             │
 │  Async generator. Yields SSE strings to the client in real time.        │
 │  Yields 3 event types: status  |  customer_reply  |  final             │
 │                                                                         │
 │  apply_business_rules()   ◄── called after every run, in all 3 paths   │
 │  Deterministic Python overrides on top of LLM output.                  │
 │  Checks DB for order existence. Enforces refund approval rules.         │
 └──────────────────────────────┬──────────────────────────────────────────┘
                                │
 AGENT LAYER  ──  agents.py     │
 ┌──────────────────────────────▼──────────────────────────────────────────┐
 │                                                                         │
 │  ┌──────────────────────────────────────────────────────────────────┐  │
 │  │  Orchestrator Agent   (gpt-4.1-mini)                             │  │
 │  │  Reads the query, decides which tools to call, combines results  │  │
 │  │  Output: FinalTriageResponse                                     │  │
 │  │                                                                  │  │
 │  │  Tools (delegate to specialist agents):                          │  │
 │  │   classify_request()       handle_support_request()              │  │
 │  │        │                           │                             │  │
 │  │        ▼                           ▼            escalate_to_manager()
 │  │  Classifier Agent         Specialist Agent           │           │  │
 │  │  ollama:llama3.2          gpt-4.1-mini               ▼           │  │
 │  │  → CustomerRequest        → FinalTriageResp.   Escalation Agent  │  │
 │  │    Result                 Tools: fetch_user_   gpt-4.1-mini      │  │
 │  │                             tier(), fetch_     → Escalation      │  │
 │  │                             order_status()       Response        │  │
 │  └──────────────────────────────────────────────────────────────────┘  │
 │                                       │                                 │
 │                              ┌────────▼────────┐                        │
 │                              │    MockDB        │                        │
 │                              │  users + orders  │                        │
 │                              └─────────────────┘                        │
 └─────────────────────────────────────────────────────────────────────────┘
```

---

## Request Flow (step by step)

Example request: `user1@gmail.com` submits `"I want a refund for order #123"` via the streaming endpoint.

```
 Client                  api.py                triage_service.py
   │                       │                          │
   │  POST /triage/stream  │                          │
   │  { email, query }     │                          │
   ├──────────────────────►│                          │
   │                       │  validate TriageRequest  │
   │                       │  create AppContext       │
   │                       ├─────────────────────────►│
   │                       │                          │  create asyncio.Queue
   │                       │                          │  define emit_status() → queue.put()
   │                       │                          │  store emit_status in ctx.on_status
   │                       │                          │
   │                       │                          │  Orchestrator Agent  (gpt-4.1-mini)
   │                       │                          ├──────────────────────────────────────►
   │                       │                          │
   │                       │   ── TOOL CALL PHASE ─────────────────────────────────────────
   │                       │                          │
   │                       │                          │  tool: classify_request()
   │                       │                          │    on_status("Classifying request...")
   │◄─ SSE {"status": "Classifying request..."}───────│
   │                       │                          │    calls Classifier Agent (ollama:llama3.2)
   │                       │                          │    returns "Category: refund"
   │                       │                          │
   │                       │                          │  tool: handle_support_request()
   │                       │                          │    on_status("Looking up account...")
   │◄─ SSE {"status": "Looking up account..."}────────│
   │                       │                          │    calls Specialist Agent (gpt-4.1-mini)
   │                       │                          │      fetch_user_tier()  → "Free"
   │                       │                          │      fetch_order_status("#123") → "WIP"
   │                       │                          │    returns FinalTriageResponse JSON
   │                       │                          │
   │                       │                          │  tool: escalate_to_manager()
   │                       │                          │    on_status("Escalating to senior rep...")
   │◄─ SSE {"status": "Escalating to senior rep..."}──│
   │                       │                          │    calls Escalation Agent (gpt-4.1-mini)
   │                       │                          │    returns EscalationResponse JSON
   │                       │                          │
   │                       │   ── STREAMING PHASE ─────────────────────────────────────────
   │                       │                          │
   │                       │                          │  Orchestrator combines results
   │                       │                          │  begins generating FinalTriageResponse tokens
   │◄─ SSE {"customer_reply": "We have received..."}──│  (one event per token batch)
   │◄─ SSE {"customer_reply": "We have received your"}│
   │◄─ SSE {"customer_reply": "We have received your refund request..."}
   │                       │                          │
   │                       │   ── FINALIZATION ────────────────────────────────────────────
   │                       │                          │
   │                       │                          │  await result.get_output()
   │                       │                          │  apply_business_rules():
   │                       │                          │    category=REFUND
   │                       │                          │    → requires_human_approval = True
   │◄─ SSE {"final": {"category": "refund",           │
   │         "requires_human_approval": true,          │
   │         "order_id": "#123",                      │
   │         "customer_reply": "...",                  │
   │         "suggested_action": "..."}}               │
   │                       │                          │
```

---

## Component Reference

### Entry Points

#### `api.py` — FastAPI HTTP Server (port 8000)

The HTTP interface. Receives JSON requests, creates a per-request `AppContext`, calls the service layer, and translates agent failures into HTTP error codes.

| Endpoint | Method | What it does |
|---|---|---|
| `/health` | GET | Returns `{"status": "OK"}`. No agent calls. Used by load balancers. |
| `/triage` | POST | Calls `run_triage_streaming()`. Returns a single JSON response (`FinalTriageResponse`). |
| `/triage/stream` | POST | Calls `run_triage_stream_events()`. Returns a `StreamingResponse` with SSE events. |

- **Request validation**: FastAPI uses `TriageRequest` (Pydantic model) to automatically validate the request body. Missing `email` or `query` → HTTP 422 before any agent code runs.
- **Error detection**: if the result has `category == "unknown"`, the orchestrator failed. The API raises `HTTPException(503)` to tell the client the service is degraded.
- **Shared database**: `db_instance = MockDB()` is created once at module load. All requests share the same database instance.

---

#### `ui.py` — NiceGUI Browser UI (port 8080)

An interactive browser interface built with NiceGUI. Connects directly to `run_triage_stream_events()` -- the same function used by the streaming API endpoint.

- **`parse_sse_event(raw)`**: strips the `data: ` prefix from each SSE string and parses the JSON payload. Returns `None` if the event is malformed.
- **Event handling**: the `on_submit` async function consumes the async generator and routes each event type to the right UI element.
  - `status` events → update the status label (shows tool call progress while waiting)
  - `customer_reply` events → silently ignored (avoids showing incomplete text)
  - `final` event → reveals the reply card and result card simultaneously
- **Status label lifecycle**: slate grey ("⚙ Starting agent...") → slate grey (tool messages) → green ("✓ Done")

---

#### `main.py` — CLI Entry Point

Used for local testing and human-in-the-loop approval. Not a server -- runs once and exits.

- **`asyncio.gather()`**: fires four triage requests simultaneously. Total wait time = the slowest request, not all four added together.
- **`IS_STREAM_RESPONSE_OUTPUT` flag**: reads from `config.py` to decide which triage function to call. Lets you switch between streaming and standard modes without touching `main.py`.
- **`refund_human_feedback()`**: pauses execution with `input()` when a refund is detected, asking an admin to approve before calling `process_refunds()`.

---

### Service Layer — `triage_service.py`

The business logic hub. All three entry points call functions in this file. No UI, no HTTP framework -- pure Python.

#### `run_triage(ctx, user_query)`

Standard (non-streaming) run. Calls `orchestrator_agent.run()`, applies business rules, logs usage, returns `FinalTriageResponse`. Used by `main.py` (non-streaming mode) and internally by `api.py /triage`.

#### `run_triage_streaming(ctx, user_query)`

Streaming run that prints partial tokens to the terminal. The streaming only applies to the final token generation phase -- tool calls still block. Returns the complete `FinalTriageResponse` when done. Used by `main.py` (streaming mode) and `api.py /triage`.

#### `run_triage_stream_events(ctx, user_query)` → `AsyncIterator[str]`

The SSE generator. Yields three types of events as strings in `data: {json}\n\n` format:

| Event key | When it's yielded | What it contains |
|---|---|---|
| `status` | During tool calls (before streaming phase) | Progress message string, e.g. `"Classifying request..."` |
| `customer_reply` | During token streaming phase | Partial `customer_reply` string, one per token batch |
| `final` | Once, after streaming completes | Full `FinalTriageResponse` as a JSON object |

The **callback + queue** pattern bridges tool calls to the generator:
1. `asyncio.Queue` is created inside this function.
2. `emit_status(message)` is defined to put messages in the queue.
3. `emit_status` is stored in `AppContext.on_status`.
4. Tools call `await ctx.deps.on_status("message")` → message lands in the queue.
5. The generator drains the queue with `while not status_queue.empty(): yield await status_queue.get()` before and inside the streaming loop.

#### `apply_business_rules(ctx, output)` → `FinalTriageResponse`

Deterministic Python overrides applied after every orchestrator run. The LLM's output is the input; a corrected `FinalTriageResponse` is the output.

| Rule | Condition | Action |
|---|---|---|
| Refund approval | `category == REFUND` | Force `requires_human_approval = True` |
| Non-existent order | `order_id` is not `None` AND `db.get_order_status()` raises `KeyError` | **Early return** with a specific `customer_reply` asking the customer to verify their order number. Overrides the entire LLM response. |
| No order | `order_id is None` | Force `requires_human_approval = False` (nothing to approve) |

The non-existent order rule **returns early** (not just sets a flag) because subsequent rules could overwrite `requires_human_approval`. An early `return FinalTriageResponse(...)` bypasses all remaining rules.

#### `process_refunds(ctx, order_id)` → `str`

Calls `db.update_order_status(order_id, "REFUNDED")`. Only called from `main.py` after admin approval. Returns a success/failure string for logging.

---

### Agent Layer — `agents.py`

Defines all four agents. Each agent's system prompt, tools, and output validator are co-located in this file.

#### Orchestrator Agent

**Model**: `gpt-4.1-mini` (OpenAI)
**Output**: `FinalTriageResponse`
**Role**: the brain of the system. Reads the customer query, decides which combination of tools to call, combines the results, and produces the final structured response.

**System prompt** (dynamic, built per request):
- Injected with `ctx.deps.user_email` at runtime
- Lists all three tools and describes the expected workflow
- Tells the orchestrator to classify first, then handle/escalate as needed

**Tools**:

| Tool | Calls | Returns |
|---|---|---|
| `classify_request(customer_message)` | Classifier Agent | `"Category: refund"` (string) |
| `handle_support_request(customer_message)` | Specialist Agent | `FinalTriageResponse` as JSON string |
| `escalate_to_manager(customer_message, reason)` | Escalation Agent | `EscalationResponse` as JSON string |

All three tools: (1) call `ctx.deps.on_status()` first for live UI feedback, (2) run the delegate agent, (3) catch all exceptions and return a descriptive error string so the orchestrator can reason about failures.

**Output validator** (`validate_orchestrator`): rejects `customer_reply` shorter than 10 characters. Skips validation on partial streaming outputs.

---

#### Classifier Agent

**Model**: `ollama:llama3.2` (local, free)
**Output**: `CustomerRequestResult` wrapped in `PromptedOutput`
**Role**: categorizes the customer's message into `refund`, `technical_support`, or `general_query`. Simple task → cheap local model.

**Why `PromptedOutput`**: small local models don't reliably follow the function-calling JSON protocol. `PromptedOutput` converts the schema into plain-text instructions in the system prompt instead, which local models handle more reliably.

**Static system prompt**: set at construction time, no per-request injection needed. The classifier has no database access and doesn't need user context.

---

#### Specialist Agent

**Model**: `gpt-4.1-mini` (OpenAI)
**Output**: `FinalTriageResponse`
**Role**: handles complex requests that need database lookups. Checks the customer's account tier and order status, then generates a detailed response.

**System prompt** (dynamic):
- Injected with `ctx.deps.user_email`
- Instructions to always check the database before responding
- Critical note: order IDs require the `#` symbol (e.g., `#123`, not `123`)

**Tools**:
- `fetch_user_tier()` — reads `user_email` from `ctx.deps`, queries `db.get_user_tier()`. No LLM parameter needed (email is already in context).
- `fetch_order_status(order_id)` — LLM provides `order_id` extracted from the user's message, queries `db.get_order_status()`.

**Output validator** (`validate_specialist_output`): rejects short `customer_reply` (< 10 chars) and vague `suggested_action` (< 5 chars).

---

#### Escalation Agent

**Model**: `gpt-4.1-mini` (OpenAI)
**Output**: `EscalationResponse`
**Role**: generates an internal escalation report for high-risk cases. Produces a severity rating, department assignment, and a detailed internal memo for the human reviewer.

**Output validator** (`validate_escalation`): rejects invalid `severity` values (must be `low`/`medium`/`high`/`critical`) and rejects `internal_memo` shorter than 20 characters.

---

### Data Models — `src/schemas.py`

Pydantic models that serve as the contract between all layers.

#### `TriageRequest`

```python
class TriageRequest(BaseModel):
    email: str
    query: str
```

FastAPI uses this to validate incoming HTTP request bodies. If either field is missing, FastAPI returns HTTP 422 automatically before any agent code runs.

---

#### `CustomerRequestResult`

```python
class CustomerRequestResult(BaseModel):
    category: RequestCategory
```

Output of the Classifier Agent. A single field: which of the four categories the request belongs to.

---

#### `FinalTriageResponse`

The main output model, produced by the Orchestrator and Specialist agents and returned to the client.

| Field | Type | Who sets it | What it means |
|---|---|---|---|
| `requires_human_approval` | `bool` | LLM (then overridden by `apply_business_rules`) | Whether a human must review before acting |
| `order_id` | `str \| None` | LLM extracts from user's message | The order referenced, if any |
| `suggested_action` | `str` | LLM | Internal instruction for the support team |
| `customer_reply` | `str` | LLM | The message to send to the customer |
| `category` | `RequestCategory` | LLM | Classified request type |

---

#### `EscalationResponse`

Internal-only. Produced by the Escalation Agent, serialized to JSON string by the `escalate_to_manager` tool, and read by the Orchestrator.

| Field | What it contains |
|---|---|
| `severity` | `"low"`, `"medium"`, `"high"`, or `"critical"` |
| `department` | `"billing"`, `"security"`, or `"management"` |
| `internal_memo` | Detailed explanation for the human reviewer |
| `customer_reply` | Holding message to send to the customer |

---

#### `RequestCategory` (Enum)

```python
class RequestCategory(str, Enum):
    REFUND           = "refund"
    TECHNICAL_SUPPORT = "technical_support"
    GENERAL_QUERY    = "general_query"
    UNKNOWN          = "unknown"   # sentinel for orchestrator failure
```

`UNKNOWN` is the error sentinel. When the orchestrator fails, the fallback `FinalTriageResponse` uses `category="unknown"`. The API layer detects this and returns HTTP 503.

---

### `AppContext` — `src/config.py`

```python
@dataclass
class AppContext:
    db: MockDB                                        # shared across all requests
    user_email: str                                   # per-request
    on_status: Callable[[str], Awaitable[None]] | None = None  # optional callback
```

The single object passed to every agent run. Carries:
- **`db`**: the shared `MockDB` instance. Created once when the server/CLI starts. All requests share it.
- **`user_email`**: set fresh for each request. Each user gets their own context even in parallel runs.
- **`on_status`**: an optional async function. When set (streaming runs only), tools call it to emit status messages. When `None` (CLI, tests), calls are safely skipped with `if ctx.deps.on_status:`.

---

### `MockDB` — `src/db.py`

In-memory database with hardcoded users and orders. In production, this would be replaced with a real database (PostgreSQL, SQLite, etc.) behind the same interface.

| Method | Parameters | Returns |
|---|---|---|
| `get_user_tier(email)` | `str` | `"Free"` or `"Premium"`. Raises `KeyError` if user not found. |
| `get_order_status(order_id)` | `str` | Status string (e.g. `"WIP"`, `"Refund Processing"`). Raises `KeyError` if order not found. |
| `update_order_status(order_id, new_status)` | `str, str` | `True` if updated, `False` if order not found. |

Seeded data:
- Users: `user1@gmail.com` (Free), `user2@gmail.com` (Premium), `user3@gmail.com` (Premium)
- Orders: `#123` (user1, WIP), `#124` (user2, Refund Processing)
- `#999` does not exist → triggers the non-existent order business rule

---

### `src/logger.py` — Shared Logger Factory

```python
logger = get_logger(__name__)   # call this at the top of every service/API file
```

Returns a named `logging.Logger` with a consistent format: `timestamp - module - level - message`. The `if not logger.handlers:` guard prevents duplicate log lines when a module is imported multiple times.

Used in: `triage_service.py`, `api.py`, `ui.py`.
Not used in: `main.py` (CLI output uses `print()` intentionally for user-facing output).

---

## Error Handling Paths

Three nested layers protect the system from cascading failures:

```
Layer 1 — Tool level (agents.py)
    Each delegate tool has try/except
    On failure → returns descriptive error string to the Orchestrator
    Example: "ERROR: Escalation failed: Connection timeout. Flag for manual review."
    The Orchestrator reads the string and decides what to do next.
         |
         v
Layer 2 — Service level (triage_service.py)
    run_triage / run_triage_streaming / run_triage_stream_events all have try/except
    On failure → returns / yields FinalTriageResponse with category="unknown"
    The "unknown" category is the error sentinel that propagates up cleanly.
         |
         v
Layer 3 — API level (api.py)
    Checks if result.category == "unknown"
    On detection → raises HTTPException(503, detail={...})
    The client receives a structured error with a human-readable message and HTTP 503.
```

---

## File Map

```
automated-ai-customer-support/
│
├── api.py                  ← HTTP interface (FastAPI, port 8000)
├── ui.py                   ← Browser interface (NiceGUI, port 8080)
├── main.py                 ← CLI interface (asyncio.gather, human-in-the-loop)
│
├── src/
│   ├── triage_service.py   ← All service logic: run_triage*, apply_business_rules, process_refunds
│   ├── agents.py           ← All agent definitions: orchestrator, classifier, specialist, escalation
│   ├── schemas.py          ← All Pydantic models: FinalTriageResponse, TriageRequest, enums
│   ├── config.py           ← AppContext dataclass, model names, token limits, feature flags
│   ├── db.py               ← MockDB (in-memory users + orders)
│   └── logger.py           ← get_logger() factory
│
├── tests/
│   ├── test_business_rules.py  ← 9 tests for apply_business_rules()
│   └── test_validators.py      ← 9 tests for agent output validators
│
└── _learning/
    ├── ARCHITECTURE.md          ← this file
    ├── PYDANTIC_AI_AGENT_GUIDE.md
    └── PYTHON_CONCEPTS.md
```

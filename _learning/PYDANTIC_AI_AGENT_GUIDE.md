# Pydantic AI Agent Development Guide

A practical, battle-tested guide to building production-grade AI agents with Pydantic AI.
This document was created from real engineering decisions made while building an automated customer support system. It serves as both a reference for humans and an instruction set for AI agents tasked with creating new agents.

---

## Table of Contents

1. [Core Philosophy](#core-philosophy)
2. [Architecture Overview](#architecture-overview)
3. [Step-by-Step: Building an Agent from Scratch](#step-by-step-building-an-agent-from-scratch)
4. [Structured Output Schemas](#structured-output-schemas)
5. [Dependency Injection with AppContext](#dependency-injection-with-appcontext)
6. [Dynamic System Prompts](#dynamic-system-prompts)
7. [Tools with @agent.tool](#tools-with-agenttool)
8. [Output Validation with ModelRetry](#output-validation-with-modelretry)
9. [Deterministic Business Rule Guardrails](#deterministic-business-rule-guardrails)
10. [Extracting Shared Business Logic](#extracting-shared-business-logic)
11. [Usage Limits and Cost Control](#usage-limits-and-cost-control)
12. [Streaming](#streaming)
13. [Multi-Agent Orchestration](#multi-agent-orchestration)
14. [Model Selection Guidelines](#model-selection-guidelines)
15. [Common Mistakes to Avoid](#common-mistakes-to-avoid)
16. [Testing Agent Systems](#testing-agent-systems)
17. [API Layer with FastAPI](#api-layer-with-fastapi)
18. [Interactive UI with NiceGUI](#interactive-ui-with-nicegui)
19. [Structured Logging](#structured-logging)
20. [Interview Prep](#interview-prep)
21. [Blueprint: Create Any Agent from a Description](#blueprint-create-any-agent-from-a-description)

---

## Core Philosophy

These principles govern every design decision in this codebase:

1. **LLMs decide, Python enforces.** Use the LLM for what it is good at: understanding natural language, interpreting intent, generating human-like responses. Enforce business rules deterministically in Python code, never in the LLM.

2. **Structured output is the contract.** The output schema is the bridge between the LLM's reasoning and your application logic. Every piece of data your code needs to act on must be a field in the output schema.

3. **Don't trust, verify.** LLMs are probabilistic. The same input can produce different outputs. Always validate LLM output against your own data sources (database, APIs) before acting on it.

4. **Dependencies flow through context.** Per-request data (user identity, session) lives in the dependency context. Shared resources (database, HTTP clients) are passed through the same context but instantiated once.

5. **Co-locate agent capabilities.** An agent's system prompt, tools, and output validator should live together in one place. Anyone reading the code should understand the agent's full capability set at a glance.

6. **Let agents delegate, let Python override.** An orchestrator agent decides which specialist agents to call. But business rules (refund approval, order validation) are always enforced by Python code on the final output, never left to the LLM's judgment.

### Why Pydantic AI?

This project uses Pydantic AI over alternatives (LangChain, LlamaIndex, raw OpenAI SDK). The reasons:

| Consideration | Pydantic AI | LangChain |
|---|---|---|
| Structured output | Native `BaseModel` -- same class you already know | Custom "output parsers" on top of strings |
| Dependency injection | `RunContext[AppContext]` -- typed, IDE-autocomplete | Callback system -- harder to trace statically |
| Abstraction level | Thin -- wraps the model API, little else | Heavy -- chains, runnables, LCEL, callbacks |
| Debugging | Read the source in minutes | Multiple library files between your code and the API |
| Learning curve | If you know Pydantic, you know 80% of this | Separate framework vocabulary to learn |

The core argument: in production, you need to understand exactly what your code does at every step. LangChain's abstraction layers make debugging harder. Pydantic AI gives you type-safe structured output -- a Pydantic `BaseModel` is the output contract -- without requiring you to learn a new paradigm on top of Python.

This doesn't mean LangChain is bad. For rapid prototyping, pre-built chains, or teams already invested in LangChain's ecosystem, it's a valid choice. For a greenfield project where control and debuggability matter most, Pydantic AI is the better fit.

---

## Architecture Overview

This project uses an **agent delegation** pattern with three entry points: an HTTP API (FastAPI), a browser UI (NiceGUI), and a CLI (main.py). All three share the same service layer. An orchestrator agent is the brain that decides which specialist agents to call. Python code enforces business rules on the final output.

```
HTTP POST /triage {"email": "...", "query": "..."}
    |
    v
[FastAPI] -- Validates request, creates AppContext, calls run_triage()
    |
    v
[Orchestrator Agent] -- Cloud LLM (OpenAI gpt-4.1-mini)
    |
    |-- tool: classify_request()
    |       --> [Classifier Agent] -- Local LLM (Ollama llama3.2) -- Cheap, fast
    |
    |-- tool: handle_support_request()
    |       --> [Specialist Agent] -- Cloud LLM (OpenAI gpt-4.1-mini)
    |               |-- tool: fetch_user_tier()
    |               |-- tool: fetch_order_status()
    |               |-- @output_validator
    |
    |-- tool: escalate_to_manager()
    |       --> [Escalation Agent] -- Cloud LLM (OpenAI gpt-4.1-mini)
    |               |-- @output_validator
    |
    Orchestrator combines all results into FinalTriageResponse
    |
    v
[apply_business_rules()] -- Python code overrides LLM decisions
    |
    |-- Refund detected? --> force requires_human_approval = True
    |-- Order doesn't exist in DB? --> force requires_human_approval = False
    |-- No order ID? --> force requires_human_approval = False
    |
    v
[Human Approval] -- If requires_human_approval --> Admin confirms
    |
    v
[Process Action] -- Database update
    |
    v
[FastAPI Response] -- HTTP 200 + FinalTriageResponse JSON
                   -- or HTTP 503 if orchestrator failed
                   -- or SSE stream via /triage/stream (status + partial + final events)
```

Key design decisions:
- **HTTP API layer**: FastAPI serves the agent system over HTTP. Pydantic models are shared between agent output and API response -- zero conversion needed. Both standard (JSON) and streaming (SSE) endpoints are available.
- **Agent delegation**: The orchestrator LLM decides which agents to call and in what order, instead of hardcoded Python `if/else` routing
- **Multiple models**: Cheap local model (Ollama) for classification, capable cloud model (OpenAI) for complex reasoning and orchestration
- **Structured output at every stage**: Classifier returns `CustomerRequestResult`, specialist returns `FinalTriageResponse`, escalation returns `EscalationResponse`
- **Business rules in Python**: Refund approval, order existence checks, and approval flags are enforced by code, never by the LLM's judgment
- **Usage roll-up**: All delegate agents pass `usage=ctx.usage` so the orchestrator tracks total cost across the entire chain
- **Layered error handling**: Delegate failures absorbed by tools (Layer 1), orchestrator failures caught by triage service (Layer 2), fallback responses converted to HTTP 503 by the API (Layer 3)
- **Status events via callbacks**: Delegate tools emit progress updates ("Classifying...", "Looking up account...") through an optional callback in `AppContext`, bridged to SSE via `asyncio.Queue`

---

## Step-by-Step: Building an Agent from Scratch

This section walks through every step needed to create a Pydantic AI agent. Follow this order.

### Step 1: Define the Output Schema

Before writing any agent code, define what the agent should return. This is your contract.

```python
from pydantic import BaseModel, Field
from enum import Enum

# Use string enums -- LLMs serialize them more reliably than integer enums
class RequestCategory(str, Enum):
    REFUND = "refund"
    TECHNICAL_SUPPORT = "technical_support"
    GENERAL_QUERY = "general_query"
    UNKNOWN = "unknown"  # Used for fallback responses when the orchestrator fails

class CustomerRequestResult(BaseModel):
    """The result of the customer request classification"""
    category: RequestCategory = Field(
        description="The classified category of the user's request"
    )
```

Rules for output schemas:
- Every field MUST have a `description` in `Field()` -- this is what the LLM reads to understand what to return
- Use `str | None` with `default=None` for optional fields (not every request has an order ID)
- Use string enums, not integer enums
- Only include fields that the LLM can uniquely provide (its interpretation, reasoning, response). Facts you can look up yourself do not belong in the schema

### Step 2: Define the Dependency Context

The context carries per-request data and shared resources to your agent's system prompt, tools, and validators.

```python
from dataclasses import dataclass
from typing import Callable, Awaitable

@dataclass
class AppContext:
    db: MockDB          # Shared resource -- one instance for all requests
    user_email: str     # Per-request data -- different for each user
    on_status: Callable[[str], Awaitable[None]] | None = None  # Optional status callback
```

Key distinction:
- **Shared resources** (database connections, HTTP clients): Created once, passed to every context
- **Per-request data** (user email, session ID): Different for each agent run, set when creating the context
- **Optional callbacks** (`on_status`): Functions that distant parts of the code can call to communicate. Set to `None` by default so existing code (main.py, tests) doesn't need to change. Only the streaming API sets a real function here.

### Callables as context fields

A **callable** in Python is anything you can call with parentheses `()`. Functions are values -- you can store them in variables, pass them as arguments, and put them in dataclasses. The `on_status` field stores an optional async function:

```python
on_status: Callable[[str], Awaitable[None]] | None = None
```

Reading this type hint:
- `Callable` -- it's a function you can call
- `[[str]]` -- it takes one argument: a `str`
- `Awaitable[None]` -- it's an `async` function (returns something you can `await`), with no meaningful return value
- `| None = None` -- optional, defaults to `None`

**Why put a function in the context?** Because agent tools and the SSE generator live in different files and can't directly communicate. Tools in `agents.py` know what's happening ("I'm classifying now") but can't send events to the client. The generator in `triage_service.py` can yield SSE events but can't see inside tool calls. The callback bridges this gap: the generator creates a function and stores it in `AppContext`. When a tool runs, it calls that function. The function puts the message somewhere the generator can pick it up. See [Status Events During Tool Calls](#status-events-during-tool-calls-callback--queue-pattern) for the full implementation.

### Step 3: Configure the Agent

```python
from pydantic_ai import Agent, RunContext, PromptedOutput, ModelRetry

specialist_agent = Agent(
    model="openai:gpt-4.1-mini",    # The LLM to use
    output_type=FinalTriageResponse, # The structured output schema
    deps_type=AppContext,            # The dependency context type
    retries=3                        # Max retries on validation failure
)
```

Parameters explained:
- `model`: The LLM provider and model name (e.g., `"ollama:llama3.2"`, `"openai:gpt-4.1-mini"`)
- `output_type`: A Pydantic BaseModel class. The agent is forced to return data matching this schema
- `deps_type`: The type of the dependency context. Enables type-safe access in system prompts and tools
- `retries`: How many times the agent can retry when the output validator raises `ModelRetry`
- Use `PromptedOutput(SchemaClass)` for models that work better with prompted JSON output (like local Ollama models)

### `PromptedOutput` vs plain schema: why Ollama needs it

OpenAI models support the **function calling API** natively. When you pass `output_type=FinalTriageResponse`, Pydantic AI sends the JSON schema to OpenAI using the structured output / function-calling protocol. The model is specifically trained to follow this protocol and return valid JSON matching the schema.

Local models (Ollama `llama3.2`) are smaller and may not reliably implement the function-calling protocol. If you pass `output_type=CustomerRequestResult` directly to an Ollama agent, the model may return plain text instead of JSON, or JSON with wrong field names, causing validation errors.

`PromptedOutput(CustomerRequestResult)` is the fix:

```python
# For cloud models: native function-calling protocol
classifier_agent = Agent(
    model="ollama:llama3.2",
    output_type=PromptedOutput(CustomerRequestResult),  # Wrap for local models
)

# For cloud models: native structured output, no wrapper needed
orchestrator_agent = Agent(
    model="openai:gpt-4.1-mini",
    output_type=FinalTriageResponse,  # Direct, no wrapper
)
```

`PromptedOutput` converts the JSON schema into instructions that appear in the system prompt: `"You must respond with a JSON object matching this schema: {schema}"`. Instead of relying on function-calling training, the model is just told to produce JSON. Most models -- even small ones -- can follow this instruction reliably.

**Rule**: always try without `PromptedOutput` first. If you get validation errors or non-JSON responses from a local model, wrap the output type with `PromptedOutput`.

### Step 4: Add a Dynamic System Prompt

Use `@agent.system_prompt` to inject runtime data from the dependency context into the prompt.

```python
@specialist_agent.system_prompt
def build_system_prompt(ctx: RunContext[AppContext]) -> str:
    return f"""
        You are a customer support specialist.
        The customer's email is: {ctx.deps.user_email}
        YOU MUST ALWAYS check the database before responding.
    """
```

Why dynamic over static:
- Static prompts (passed as `system_prompt="..."` in the constructor) cannot access runtime data
- Dynamic prompts have access to `RunContext` which carries the dependency context
- This lets you inject user-specific information (email, tier, history) at runtime

### Step 5: Add Tools with @agent.tool

Tools give the agent the ability to perform actions and retrieve data.

```python
@specialist_agent.tool
def fetch_user_tier(ctx: RunContext[AppContext]) -> str:
    """Fetch the user tier level from the database using the customer's email."""
    try:
        return ctx.deps.db.get_user_tier(ctx.deps.user_email)
    except KeyError:
        return "User not found in the database."

@specialist_agent.tool
def fetch_order_status(ctx: RunContext[AppContext], order_id: str) -> str:
    """Fetch the order status from the database using the provided order ID."""
    try:
        return ctx.deps.db.get_order_status(order_id)
    except KeyError:
        return "Order ID not found in the database."
```

Rules for tools:
- The **docstring** is sent to the LLM as the tool description. Write it clearly
- Parameters (other than `ctx`) become the tool's input schema -- the LLM decides what values to pass
- If a value is already known (like the user email), pull it from `ctx.deps` instead of making the LLM provide it. One less thing for the model to get wrong
- Always handle errors gracefully -- return a string error message, don't raise exceptions. The LLM needs to understand what went wrong
- Use `@agent.tool` (with context) for tools that need dependency access
- Use `@agent.tool_plain` for tools that don't need any context

### Step 6: Add Output Validation

The output validator checks the quality of the LLM's response after it returns structured data.

CRITICAL: If the agent will ever be used with `run_stream()`, the output validator MUST guard against partial outputs. During streaming, the validator is called multiple times -- once for each partial output as tokens arrive, and once for the final complete output. Early partial outputs will have empty or incomplete fields. Without the `ctx.partial_output` guard, the validator will raise `ModelRetry` on incomplete data and kill the stream before the model finishes generating.

```python
@specialist_agent.output_validator
def validate_output(ctx: RunContext[AppContext], output: FinalTriageResponse) -> FinalTriageResponse:
    # REQUIRED: Skip validation for partial streaming outputs -- data is still arriving
    # ctx.partial_output is True when the validator is called during streaming on incomplete data
    # ctx.partial_output is False when it's the final complete output
    if ctx.partial_output:
        return output

    if not output.customer_reply or len(output.customer_reply.strip()) < 10:
        raise ModelRetry(
            "customer_reply is too short or empty. Provide a meaningful, professional response."
        )
    if not output.suggested_action or len(output.suggested_action.strip()) < 5:
        raise ModelRetry(
            "suggested_action is too vague. Describe the specific action to take."
        )
    return output
```

When to use `ModelRetry` vs deterministic override:

| Situation | Approach |
|---|---|
| Output is low quality (vague, too short, missing detail) | `ModelRetry` in output validator |
| Output violates a business rule (wrong approval flag, invalid state) | Deterministic override in application code |

`ModelRetry` re-prompts the model with your error message. Use it when the model CAN fix the problem.
Deterministic overrides silently correct the output. Use them when only your code knows the right answer.

### What `ctx` provides in the output validator

The `ctx` parameter (`RunContext[AppContext]`) carries metadata about the current state of the agent run:

- **`ctx.deps`** -- your `AppContext` (database, user email). Same as in tools and system prompts.
- **`ctx.partial_output`** -- boolean. `True` during streaming when data is still arriving, `False` on the final complete output. ALWAYS check this first in validators.
- **`ctx.usage`** -- current token usage for this run so far.
- **`ctx.messages`** -- the conversation messages exchanged so far.
- **`ctx.run_id`** -- unique identifier for this specific agent run.

### Deep dive: How `ctx.partial_output` works during streaming

When you call `agent.run()` (standard), the output validator runs **once** on the complete output. Simple.

When you call `agent.run_stream()`, the output validator is called **many times** as tokens arrive:

```
Token 1 arrives:  {"requires_human_approval": true}
    --> validator called with ctx.partial_output = True
    --> customer_reply is empty (hasn't arrived yet)
    --> WITHOUT the guard: validator raises ModelRetry("customer_reply too short") --> STREAM KILLED
    --> WITH the guard: validator returns output immediately --> stream continues

Token 5 arrives:  {"requires_human_approval": true, "customer_reply": "We have rec"}
    --> validator called with ctx.partial_output = True
    --> customer_reply exists but is incomplete
    --> guard skips validation --> stream continues

Final token:      {"requires_human_approval": true, "customer_reply": "We have received your refund request..."}
    --> validator called with ctx.partial_output = False (this is the final output)
    --> NOW full validation runs: length checks, content checks, etc.
    --> If validation fails: ModelRetry triggers a full retry
```

This is why `if ctx.partial_output: return output` must be the **first line** of every output validator. Without it, streaming is impossible for any agent with output validation -- the validator will reject every partial output because fields haven't fully materialized yet.

**Key insight**: `ctx.partial_output` is how you can tell whether the model is still generating or done. You don't need a separate flag or state machine. Pydantic AI manages the streaming lifecycle and tells your validator exactly where it is in the process.

### Step 7: Run the Agent

There are two ways to run an agent: standard (blocking) and streaming.

**Standard run** -- waits for the full response, then returns it:

```python
from pydantic_ai import UsageLimits

specialist_response = await specialist_agent.run(
    user_prompt=user_query,
    deps=AppContext(db=db_instance, user_email="user@example.com"),
    usage_limits=UsageLimits(request_limit=10, total_tokens_limit=4000)
)

# Access the structured output directly via .output
print(specialist_response.output.customer_reply)
print(specialist_response.output.order_id)

# Track usage for cost monitoring
print(specialist_response.usage())
```

**Streaming run** -- tokens arrive as they are generated:

```python
async with orchestrator_agent.run_stream(
    user_prompt=user_query,
    deps=AppContext(db=db_instance, user_email="user@example.com"),
    usage_limits=UsageLimits(request_limit=10, total_tokens_limit=4000)
) as result:
    # Stream partial structured output as it arrives
    async for partial_output in result.stream_output():
        if partial_output.customer_reply:
            print(f"\r{partial_output.customer_reply}", end="", flush=True)

    print()

    # IMPORTANT: Use await result.get_output(), NOT result.output
    # run_stream() returns a stream that might still be in progress
    output = await result.get_output()

    # Only call usage() after get_output() -- otherwise numbers are incomplete
    print(result.usage())
```

See the [Streaming](#streaming) section for full details on streaming patterns and pitfalls.

---

## Structured Output Schemas

The output schema is the most important design decision in your agent. It defines what the LLM returns and what your application code can act on.

### Example: FinalTriageResponse

```python
class FinalTriageResponse(BaseModel):
    """The final response to the customer"""
    requires_human_approval: bool = Field(
        description="Set to True if the customer's request is related to a refund, return, or modifying account security. Otherwise, set to False."
    )
    order_id: str | None = Field(
        default=None,
        description="The order ID related to the request, if applicable. Must include the '#' symbol."
    )
    suggested_action: str = Field(
        description="How should you respond to the customer's request?"
    )
    customer_reply: str = Field(
        description="A single concise sentence to say to the customer. Must be friendly and professional."
    )
    category: RequestCategory = Field(
        description="The classified category of the user's request"
    )
```

### Example: EscalationResponse

Used by the escalation agent for high-risk cases that require human review:

```python
class EscalationResponse(BaseModel):
    """Internal escalation report for high-risk customer cases"""
    severity: str = Field(description="Severity level: 'low', 'medium', 'high', or 'critical'")
    department: str = Field(description="Which department should handle this: 'billing', 'security', or 'management'")
    internal_memo: str = Field(description="A detailed internal summary for the human reviewer explaining why this case was escalated and what action is recommended.")
    customer_reply: str = Field(description="A professional message to send the customer while their case is being reviewed.")
```

Note: Different agents can have different output schemas. The orchestrator returns `FinalTriageResponse`, the escalation agent returns `EscalationResponse`. Each schema is tailored to what that specific agent needs to communicate.

### Example: TriageRequest (API input)

Pydantic models serve double duty. They define agent output schemas AND API request validation:

```python
class TriageRequest(BaseModel):
    """The API request input to the triage agent"""
    email: str
    query: str
```

FastAPI validates incoming HTTP requests against this model automatically. If `email` or `query` is missing, FastAPI returns HTTP 422 before your agent code runs. The same Pydantic validation you use for agent output now protects your API inputs.

### Design principles

1. **Every field the application needs must be in the schema.** If your refund logic needs an order ID, the schema must include `order_id`. The LLM identifies it from the conversation; your code uses it to process the refund.

2. **Field descriptions are instructions to the LLM.** The `description` parameter in `Field()` is sent directly to the model as part of the JSON schema. The LLM reads these descriptions to understand what value to produce for each field. Write them like you are instructing a person: be specific about format, constraints, and expectations. A vague description like `"The order"` will get vague results. A precise description like `"The order ID related to the customer's request, if applicable. Must include the '#' symbol, e.g. '#123'."` tells the LLM exactly what format to use.

3. **Use `str | None` for conditional fields.** Not every request involves an order. Making `order_id` optional with `default=None` lets the LLM skip it for general queries.

4. **Use string enums, not integer enums.** LLMs generate text. Asking an LLM to produce `"refund"` is far more natural and reliable than asking it to produce `0`. String enums (`class RequestCategory(str, Enum)`) serialize directly into human-readable JSON that the model already understands. Integer enums require the model to memorize an arbitrary mapping between numbers and meanings -- one more thing to get wrong.

5. **Don't include fields you can verify yourself.** If you can check whether an order exists by querying your database, don't add `is_order_found: bool` to the schema. The LLM might say the order exists when it doesn't, or vice versa. Only the database knows the truth. Every field in the schema should be something that **only the LLM can provide** -- its interpretation of the user's intent, its natural language response, the entity it extracted from free text. If your Python code can determine the answer deterministically, keep it out of the schema and verify it in code.

6. **Include fields even if you override them.** The `requires_human_approval` field is overridden by `apply_business_rules()` in many cases. So why include it? Because the LLM provides its **best judgment as a starting point**. For cases that don't hit any of your explicit business rules, the LLM's judgment is the fallback. Python code only overrides when a specific business rule disagrees. If you removed the field entirely, you'd have no default value for edge cases your rules don't cover.

---

## Dependency Injection with AppContext

Dependencies are the data and services your agent needs at runtime. They flow through `RunContext` to system prompts, tools, and validators.

```python
@dataclass
class AppContext:
    db: MockDB          # Shared: one database for all users
    user_email: str     # Per-request: each user has their own email
```

### How it flows

```
main.py creates AppContext(db=shared_db, user_email="user1@gmail.com")
    |
    v
agent.run(deps=ctx)
    |
    |-- @agent.system_prompt receives RunContext[AppContext]
    |       --> accesses ctx.deps.user_email to personalize the prompt
    |
    |-- @agent.tool receives RunContext[AppContext]
    |       --> accesses ctx.deps.db to query the database
    |
    |-- @agent.output_validator receives RunContext[AppContext]
            --> can access any dependency for validation
```

### Why not pass data as part of the prompt?

You COULD put the user email in the user prompt: `"User email: user1@gmail.com. Query: I want a refund."`
But this mixes data with the user's message. Dependency injection keeps them separate:
- The system prompt gets the email from `ctx.deps` -- clean, reliable, type-safe
- The user prompt stays exactly what the user said -- no data injection, no formatting hacks
- Tools access the DB through `ctx.deps` -- no globals, no module-level state

### Why `dataclass` for AppContext instead of Pydantic `BaseModel`?

`AppContext` is defined with `@dataclass`, not `class AppContext(BaseModel)`. This is intentional:

- **`BaseModel`** is for data that comes from outside your code (HTTP requests, LLM output, JSON files). It validates and coerces values because you can't trust external input.
- **`dataclass`** is for internal data structures that your own trusted code creates and controls. It's a lightweight container with no validation overhead.

`AppContext` is always created by your Python code (`AppContext(db=db_instance, user_email=email)`). The database instance and the email string are set by code you control. There is nothing to validate. Adding Pydantic validation here would add overhead with zero safety benefit.

**The rule**: if data arrives from the outside world, use `BaseModel`. If data is created by your own code, use `dataclass`.

---

## Dynamic System Prompts

Static system prompts are set once at agent creation. Dynamic system prompts are generated at runtime using dependency data.

```python
# Static: passed in constructor (cannot use runtime data)
classifier_agent = Agent(
    model="ollama:llama3.2",
    system_prompt="You are a triage expert. Categorize customer messages."
)

# Dynamic: uses @agent.system_prompt decorator (has access to RunContext)
@specialist_agent.system_prompt
def build_prompt(ctx: RunContext[AppContext]) -> str:
    return f"""
        You are a customer support specialist.
        The customer's email is: {ctx.deps.user_email}
    """
```

Use static prompts for agents with no per-request variation (like the classifier).
Use dynamic prompts when the agent needs runtime data (user identity, session state, feature flags).

---

## Tools with @agent.tool

### Agent-specific vs shared tools

```python
# Agent-specific tool: use decorator -- lives with the agent definition
@specialist_agent.tool
def fetch_order_status(ctx: RunContext[AppContext], order_id: str) -> str:
    """Fetch the order status from the database."""
    ...

# Shared tool: pass in constructor -- can be reused across agents
def get_current_time() -> str:
    """Returns the current UTC time."""
    return datetime.utcnow().isoformat()

agent_a = Agent(model="...", tools=[get_current_time])
agent_b = Agent(model="...", tools=[get_current_time])
```

### Reducing LLM errors in tool calls

Every parameter a tool exposes to the LLM is a **decision the LLM must make**. Each decision is an opportunity for error. If the answer is already known (stored in context, derivable from other data), remove it from the tool's parameter list and pull it from `ctx.deps` instead.

```python
# BAD: LLM must guess/find the email to pass
@specialist_agent.tool
def fetch_user_tier(ctx: RunContext[AppContext], email: str) -> str:
    return ctx.deps.db.get_user_tier(email)

# GOOD: Email comes from context, LLM just calls the tool
@specialist_agent.tool
def fetch_user_tier(ctx: RunContext[AppContext]) -> str:
    return ctx.deps.db.get_user_tier(ctx.deps.user_email)
```

Why this matters:
- In the BAD version, the LLM must figure out the user's email from the conversation, then type it correctly as a parameter. It might hallucinate an email, use the wrong format, or pull the wrong email from a multi-user conversation.
- In the GOOD version, the email is already in `ctx.deps.user_email` -- set by your Python code when creating the `AppContext`. The LLM just calls the tool with zero parameters. There is literally nothing for it to get wrong.

**The principle: minimize the LLM's decision surface.** Only expose parameters the LLM genuinely needs to decide (like `order_id` in `fetch_order_status`, which the LLM extracts from the user's message). Everything else should come from context.

### Tool docstrings are LLM instructions

The docstring of a tool function is not just for human developers -- it is sent to the LLM as the tool's description. The model reads this description to decide **when** and **how** to call each tool.

```python
@specialist_agent.tool
def fetch_order_status(ctx: RunContext[AppContext], order_id: str) -> str:
    """Fetch the order status from the database using the provided order ID.
    If the order is not found, return 'Order ID could not be found.'"""
```

The LLM sees this description alongside all other tool descriptions and uses it to decide which tool to call. A vague docstring like `"""Get status"""` gives the LLM little information to work with. A specific docstring like the example above tells the model exactly what the tool does, what it needs, and what it returns on failure.

### Tool error handling

Tools should return string error messages, not raise exceptions. The LLM needs to understand what went wrong so it can adjust its approach:

```python
# BAD: Raises an exception -- the LLM gets no useful feedback
@agent.tool
def fetch_order(ctx: RunContext[AppContext], order_id: str) -> str:
    return ctx.deps.db.get_order_status(order_id)  # Raises KeyError if not found

# GOOD: Returns a descriptive error string -- the LLM can react
@agent.tool
def fetch_order(ctx: RunContext[AppContext], order_id: str) -> str:
    try:
        return ctx.deps.db.get_order_status(order_id)
    except KeyError:
        return "Order ID could not be found."
```

If a tool raises an exception, the agent run might crash. If it returns an error string, the LLM can read the error and try a different approach (e.g., ask the user for clarification, or skip that step).

---

## Output Validation with ModelRetry

Output validators run after the LLM returns structured data. They check quality and can ask the model to try again.

```python
from pydantic_ai import ModelRetry

@specialist_agent.output_validator
def validate_output(ctx: RunContext[AppContext], output: FinalTriageResponse) -> FinalTriageResponse:
    if not output.customer_reply or len(output.customer_reply.strip()) < 10:
        raise ModelRetry("customer_reply is too short. Provide a meaningful response.")
    return output
```

### How the ModelRetry feedback loop works

The output validator creates a conversation between your code and the LLM:

```
Step 1: LLM generates JSON output
Step 2: Pydantic parses the JSON against the schema (catches type errors, missing fields)
Step 3: Your output validator function runs (catches quality issues)
Step 4: If validator raises ModelRetry("customer_reply is too short"):
        --> Pydantic AI takes your error message
        --> Sends it BACK to the LLM as feedback in the conversation
        --> The LLM sees: "Your previous output was rejected: customer_reply is too short"
        --> The LLM tries again with a better response
Step 5: Steps 2-4 repeat up to the `retries` limit on the agent
Step 6: If all retries fail, an exception is raised to the caller
```

This is why the error message you write in `ModelRetry(...)` matters -- the LLM literally reads it. Write it like feedback to a person: tell the model exactly what was wrong and what you expect instead.

### When to use ModelRetry vs deterministic override

Use `ModelRetry` for problems the model CAN fix: vague responses, missing detail, wrong format. The model reads your feedback and adjusts its output.

Do NOT use `ModelRetry` for business rule violations. If the LLM sets `requires_human_approval=False` for a refund, re-prompting won't help -- the LLM made a judgment call based on context, and it might make the same call again. Override it deterministically in Python code instead (see next section).

| Situation | Approach | Why |
|---|---|---|
| Output is low quality (vague, too short) | `ModelRetry` in output validator | The model CAN produce a better response if told what was wrong |
| Output violates a business rule | Deterministic override in application code | Only your code knows the business rule; the model can't reliably enforce it |
| Output has wrong format | `ModelRetry` in output validator | The model CAN fix formatting when given specific feedback |
| Output contains a factual claim you can verify | Deterministic check in application code | Don't ask the model to verify facts -- check your database yourself |
| LLM writes a response inconsistent with a DB lookup result | Deterministic override in application code | The LLM will re-extract the same data from the user's message on every retry |

### Deep dive: The ModelRetry exhaustion trap

A `ModelRetry` loop will exhaust all retries (silently crashing the run) if the feedback message does not instruct the LLM to change the **specific field** that keeps triggering the rejection.

**The scenario that exposed this:** A customer asks for a refund on order `#99990`, which doesn't exist in the database. The orchestrator validator detects this and raises `ModelRetry` telling the LLM to acknowledge the order doesn't exist. On every retry, the LLM still sets `order_id="#99990"` -- because that's what the user said, and the schema says "include the order ID if the request is related to one." The validator fires again. After `MAX_RETRIES` attempts, Pydantic AI raises an exception, the outer `except` catches it, and the hardcoded fallback message appears: `"I'm sorry, I'm unable to process your request."` -- not the helpful "order not found" message you wanted.

```
ModelRetry fires: "order #99990 not found, update customer_reply"
  ↓
LLM retries -- still sets order_id="#99990" (it's in the user's message)
  ↓
Validator fires again for the same reason
  ↓
After MAX_RETRIES=3: Pydantic AI raises UnexpectedModelBehavior
  ↓
except Exception catches it → hardcoded fallback fires
  ↓
Customer sees: "I'm sorry, I'm unable to process your request."
```

**Why does this happen?** The LLM has two instructions that conflict:
1. Your `ModelRetry` message: "set order_id to null"
2. The schema field description: "The order ID related to the customer's request, if applicable"

The user explicitly mentioned `#99990` in their message. The LLM correctly determines "this request IS related to order #99990" and keeps including it -- following the schema description, which it treats as a structural contract. Your retry instruction conflicts with the schema, and the schema wins.

**The rule:** `ModelRetry` works when you are asking the LLM to **improve output quality** (write more, be more specific, change tone). It fails when you are asking the LLM to **contradict data it correctly extracted from the user's message**. For the second case, use a deterministic override after the agent returns.

**How to detect you're in this trap:** If you see the hardcoded fallback message (`"I'm sorry, I'm unable to process your request."`) when you expected a specific message from your validator, the retry loop has exhausted. Check your logs: `logger.error("Orchestrator failed: ...")` will show the `UnexpectedModelBehavior` or `UsageLimitExceeded` exception that triggered the fallback.

---

## Deterministic Business Rule Guardrails

LLMs are probabilistic. They interpret context and make judgment calls. Sometimes those calls don't match your business rules. The solution: override in Python.

### Example: Refund approval

The orchestrator's output includes a `category` field (classified via the delegate classifier agent). The LLM might set `requires_human_approval=False` because the order status says "Refund Processing" and it interprets that as "already handled." Your business rule says every refund needs human approval, regardless.

```python
requires_human_approval = output.requires_human_approval

# Business rule: refund category always requires approval
if output.category == RequestCategory.REFUND:
    requires_human_approval = True

# Business rule: if the order doesn't exist, override the entire response
# The LLM hallucinated a response implying the order is real -- Python corrects it
if output.order_id is not None:
    try:
        ctx.db.get_order_status(output.order_id)
    except KeyError:
        # Early return: the order doesn't exist, override customer_reply entirely
        return FinalTriageResponse(
            requires_human_approval=False,
            order_id=output.order_id,   # keep for logging -- shows what was attempted
            category=output.category,
            suggested_action="Order not found. Ask customer to verify order number.",
            customer_reply=f"We couldn't find order {output.order_id} in our system — "
                           "please double-check your order number and contact us if you need help."
        )

# Business rule: no order = nothing to approve
if output.order_id is None:
    requires_human_approval = False
```

### The principle

Don't ask the LLM to tell you something you can verify yourself:
- The LLM says "order #999 needs a refund" --> You check the database: #999 doesn't exist --> Override the entire response with "order not found" message
- The LLM says "no approval needed" for a refund --> Your business rule says refunds always need approval --> Override `requires_human_approval = True`

**LLM provides**: intent interpretation, natural language responses, entity extraction
**Python enforces**: business rules, data validation, access control

### Why the non-existent order rule returns early (not just sets a flag)

The initial implementation of the non-existent order rule only set `requires_human_approval = False`. This had a subtle bug: for a **refund request** on a non-existent order, the rule execution order was:

```
Rule 1: category == REFUND  →  requires_human_approval = True
Rule 2: order #99990 not found  →  requires_human_approval = False  (overwrites Rule 1!)
Final: requires_human_approval = False  ← wrong
```

The last rule to write the flag wins. This made a refund request on a non-existent order appear to not need human approval, which is the opposite of safe behavior.

The fix is to return a complete new `FinalTriageResponse` early when an order doesn't exist -- not just flip a flag. This also fixes a second problem: the LLM's `customer_reply` was hallucinating ("our billing team is reviewing it") despite the specialist's DB tool returning "Order ID could not be found." An early return lets Python write the exact reply the customer should see, bypassing the LLM's hallucination entirely.

The `order_id` is preserved in the override response so logs show what order number was attempted. This is important for debugging and audit trails.

### Why business rules go AFTER the agent, not inside it

There are three places you could try to enforce business rules. Only one is correct:

**Option 1: In the system prompt** (WRONG)
```
"You MUST set requires_human_approval to True for all refund requests."
```
Problem: The LLM might ignore this. It's a suggestion, not a guarantee. The LLM interprets the order status as "already processing" and decides no approval is needed. System prompts guide behavior -- they don't enforce it.

**Option 2: In the output validator with ModelRetry** (WRONG)
```python
@agent.output_validator
def validate(ctx, output):
    if output.category == "refund" and not output.requires_human_approval:
        raise ModelRetry("Refunds always require human approval. Set requires_human_approval to True.")
```
Problem: `ModelRetry` asks the LLM to fix the problem. The LLM made a judgment call based on its interpretation of the context. When you tell it "set this to True," it might comply -- or it might reason again that the order is already being refunded and set it back to False. You're in a non-deterministic loop. You might burn all your retries and still not get the right answer, because you're asking a probabilistic system to produce a deterministic result.

**Option 3: In Python code AFTER the agent returns** (CORRECT)
```python
output = apply_business_rules(ctx, result.output)
# Inside: if output.category == RequestCategory.REFUND: requires_human_approval = True
```
This is deterministic. It runs once. It always produces the correct result. The LLM provides its best judgment, and Python code silently corrects it where business rules disagree. No retries, no wasted tokens, no non-determinism.

### Why we still ask the LLM for `requires_human_approval`

If Python overrides the value anyway, why include it in the schema at all?

Because the LLM's judgment is the **default fallback**. Your business rules only cover cases you've explicitly coded for (refunds, missing orders). For every other case -- a customer asking to change their shipping address, or requesting account deletion, or reporting fraud -- the LLM's judgment is the only signal you have. By including the field, you get the LLM's assessment for free on every request. Your Python code only overrides the specific cases where you know better.

Think of it as a **layered decision**:
1. LLM makes its best judgment call for ALL requests (broad coverage)
2. Python code overrides specific cases where business rules exist (precise enforcement)
3. For cases without explicit rules, the LLM's judgment stands (graceful fallback)

---

## Extracting Shared Business Logic

When you have multiple ways to run the same agent (standard and streaming), the business rules must be identical. Duplicating business rules across functions is a liability -- if you update one and forget the other, you have inconsistent behavior.

Extract shared logic into a dedicated function:

```python
def apply_business_rules(ctx: AppContext, output: FinalTriageResponse) -> FinalTriageResponse:
    requires_human_approval = output.requires_human_approval

    # Business rule: refunds always require human approval
    if output.category == RequestCategory.REFUND:
        requires_human_approval = True

    # Business rule: if the order doesn't exist, override the response entirely
    if output.order_id is not None:
        try:
            ctx.db.get_order_status(output.order_id)
        except KeyError:
            return FinalTriageResponse(
                requires_human_approval=False,
                order_id=output.order_id,
                category=output.category,
                suggested_action="Order not found. Ask customer to verify order number.",
                customer_reply=f"We couldn't find order {output.order_id} in our system — "
                               "please double-check your order number and contact us if you need help."
            )

    # Business rule: no order = nothing to approve
    if output.order_id is None:
        requires_human_approval = False

    return FinalTriageResponse(
        requires_human_approval=requires_human_approval,
        order_id=output.order_id,
        category=output.category,
        suggested_action=output.suggested_action,
        customer_reply=output.customer_reply
    )
```

Note: With the orchestrator pattern, the `category` is now part of the `FinalTriageResponse` output itself (the orchestrator classifies via a tool and includes the category in its final response). This means `apply_business_rules` no longer needs a separate `intent` parameter -- it reads `output.category` directly.

Both `run_triage()` and `run_triage_streaming()` call this same function:

```python
# In run_triage (standard):
output = apply_business_rules(ctx, result.output)

# In run_triage_streaming (streaming):
output = apply_business_rules(ctx, output)
```

### The rule

Any logic that applies regardless of HOW the agent was run (standard vs streaming) belongs in a shared function. Business rules, data validation, and post-processing are all candidates for extraction. The run mode (standard vs streaming) is a transport concern -- it should never affect business decisions.

---

## Usage Limits and Cost Control

Every agent run should have cost guardrails.

```python
from pydantic_ai import UsageLimits

result = await agent.run(
    user_prompt=query,
    deps=ctx,
    usage_limits=UsageLimits(
        request_limit=10,           # Max LLM round-trips (tool calls + retries)
        total_tokens_limit=4000     # Max total tokens (input + output)
    )
)

# Always track usage
print(result.usage())
# RunUsage(input_tokens=2189, output_tokens=222, requests=4, tool_calls=4)
```

### Understanding token costs in agentic systems (token compounding)

**The fundamental reality: LLMs are stateless.** An LLM has no memory between requests. Every single request must include the ENTIRE conversation history -- system prompt, all previous messages, all previous tool calls, all previous tool results. The LLM reads everything from scratch every time.

This means token usage **compounds** with every round-trip:

```
Request 1: system prompt + user query                                    = ~300 input tokens
Request 2: system prompt + user query + tool_call_1 + tool_result_1      = ~600 input tokens
Request 3: all of the above + tool_call_2 + tool_result_2                = ~900 input tokens
Request 4: all of the above + structured output attempt                  = ~1200 input tokens
                                                                    TOTAL: ~3000 input tokens
```

This is why the specialist agent uses ~2000 input tokens across 4 requests even though the system prompt is only ~300 tokens. It is not a bug, it is not a misconfiguration -- it is how LLMs fundamentally work. Every tool call adds to the conversation history, and every subsequent request resends that growing history.

**Implications for agent design:**
- Every tool you add increases the potential number of round-trips
- Long tool results (like large JSON responses) bloat every subsequent request
- The orchestrator pattern amplifies this: the orchestrator's tool calls include the delegate agents' full results, which are resent in every subsequent orchestrator request
- Always set `total_tokens_limit` to account for compounding, not just a single request

### Model selection impacts cost

- **Reasoning models** (gpt-5-nano, o-series): Generate internal chain-of-thought tokens called `reasoning_tokens`. These are tokens the model produces while "thinking" -- they don't appear in your output but they count toward your bill. A simple one-sentence customer reply can burn 1700+ reasoning tokens on internal deliberation before producing 50 output tokens. You can see this in `result.usage()` under the `details` field: `'reasoning_tokens': 1728`.
- **Non-reasoning models** (gpt-4.1-mini, gpt-4o-mini): No reasoning token overhead. The model generates output tokens directly. Use these for tasks that don't require deep multi-step thinking.

**How to detect the problem:** Run your agent and check `result.usage()`. If you see a large `reasoning_tokens` value relative to the actual output, you're paying for thinking the task doesn't need. Switch to a non-reasoning model.

```python
# Reasoning model output (expensive):
# RunUsage(input_tokens=1038, output_tokens=1964, details={'reasoning_tokens': 1728}, requests=2)
# --> 1728 of the 1964 output tokens were just internal thinking!

# Non-reasoning model output (efficient):
# RunUsage(input_tokens=2189, output_tokens=222, details={'reasoning_tokens': 0}, requests=4)
# --> All 222 output tokens are actual useful content
```

---

## Streaming

Streaming sends tokens back to the user as they are generated, instead of waiting for the full response. This is essential for any user-facing application.

### `run()` vs `run_stream()` -- Key Differences

| Aspect | `run()` | `run_stream()` |
|---|---|---|
| Returns | `AgentRunResult` | Async context manager yielding `StreamedRunResult` |
| Access output | `result.output` (direct attribute) | `await result.get_output()` (must await) |
| Output validator | Called once on complete output | Called multiple times on partial + final output |
| Token delivery | All at once after completion | Progressive as tokens arrive |

### Streaming Structured Output

For agents with structured `output_type` (Pydantic models), use `stream_output()` to receive partial model instances:

```python
async with orchestrator_agent.run_stream(
    user_prompt=user_query,
    deps=ctx,
    usage_limits=UsageLimits(request_limit=10, total_tokens_limit=4000)
) as result:
    async for partial_output in result.stream_output():
        if partial_output.customer_reply:
            print(f"\r{partial_output.customer_reply}", end="", flush=True)

    print()
    output = await result.get_output()
    print(f"[Usage] {result.usage()}")
```

Note: Streaming works with the orchestrator agent just like any other agent. The orchestrator will call its delegate tools (classifier, specialist, escalation) and then stream its final `FinalTriageResponse` back to you. The delegate calls happen during the stream but aren't directly visible -- you only see the final structured output as it generates.

### Streaming Plain Text

For agents that return plain text (no structured `output_type`, or `output_type=str`), use `stream_text()`:

```python
async with agent.run_stream(user_prompt=query, deps=ctx) as result:
    async for text in result.stream_text():
        print(text, end="", flush=True)
```

For structured output agents, `stream_text()` will give you raw JSON fragments, which is not useful to display. Use `stream_output()` instead.

### CRITICAL RULES for Streaming

**Rule 1: Always guard output validators with `ctx.partial_output`.**

During streaming, the output validator is called on every partial output. Early partials will have empty fields. Without this guard, the validator raises `ModelRetry` before the model finishes generating.

```python
@agent.output_validator
def validate(ctx: RunContext[AppContext], output: MyOutput) -> MyOutput:
    if ctx.partial_output:
        return output  # ALWAYS skip validation for partial outputs
    # ... validate final output only ...
    return output
```

**Rule 2: Use `await result.get_output()`, not `result.output`.**

`run()` returns a completed result so `.output` works directly.
`run_stream()` returns a stream in progress. You must explicitly await the final output with `get_output()`.

```python
# Standard run:
result = await agent.run(...)
output = result.output  # Works directly

# Streaming run:
async with agent.run_stream(...) as result:
    ...
    output = await result.get_output()  # Must await
```

**Rule 3: Call `result.usage()` only after `get_output()`.**

Usage numbers are incomplete until the stream finishes and the final output is resolved.

**Rule 4: Don't mix streaming with `asyncio.gather` carelessly.**

Multiple streams running concurrently will interleave their print output in the terminal, making debugging impossible. When using streaming, either:
- Run requests sequentially for clear output
- Or capture streamed output into buffers instead of printing directly

**Rule 5: Business logic is identical between standard and streaming.**

Extract business rules into a shared function that both `run_triage()` and `run_triage_streaming()` call. The streaming mode only changes HOW tokens are delivered, never WHAT business rules apply.

### Architecture: Standard vs Streaming Triage

With the orchestrator pattern, both modes call the same orchestrator agent. The only difference is how tokens are delivered:

```
run_triage()                          run_triage_streaming()
    |                                     |
    v                                     v
orchestrator_agent.run()              orchestrator_agent.run_stream()
    |                                     |
    | (internally calls delegates:        | (internally calls delegates:
    |  classifier, specialist,            |  classifier, specialist,
    |  escalation via tools)              |  escalation via tools)
    |                                     |
    v                                     v
result.output                         stream_output() -> partial prints
    |                                 await result.get_output()
    |                                     |
    v                                     v
apply_business_rules()                apply_business_rules()    [SAME function, shared]
    |                                     |
    v                                     v
return FinalTriageResponse            return FinalTriageResponse
```

### Configurable Run Mode

Use a config flag to switch between standard and streaming without changing application logic:

```python
# config.py
IS_STREAM_RESPONSE_OUTPUT = True

# main.py
triage_func = run_triage_streaming if IS_STREAM_RESPONSE_OUTPUT else run_triage
results = await asyncio.gather(
    triage_func(AppContext(db=db, user_email=user["email"]), user["query"]),
    ...
)
```

### Status Events During Tool Calls (Callback + Queue Pattern)

With the orchestrator pattern, the LLM performs all tool calls (classify, specialist, escalate) BEFORE it begins streaming its final output. This means the user stares at a blank screen while the real work happens. Status events solve this by sending updates like "Classifying your request..." and "Looking up your account..." during the tool call phase.

**The problem:** The tools in `agents.py` know what's happening, but they're regular async functions -- they can't `yield` SSE events. The async generator in `triage_service.py` can yield SSE events, but it can't see inside the tools. These are two distant parts of the code that need to communicate.

**The solution:** A **callback function** stored in `AppContext`, plus an `asyncio.Queue` to bridge the timing gap.

#### Step 1: Add a status callback to AppContext

```python
@dataclass
class AppContext:
    db: MockDB
    user_email: str
    on_status: Callable[[str], Awaitable[None]] | None = None
```

`on_status` is an optional async function. When `None` (main.py, tests, non-streaming API), tools skip status events entirely. When set (streaming API), tools call it to emit status updates. The `None` default keeps all existing code backward-compatible.

#### Step 2: Tools call the callback

Each delegate tool in `agents.py` calls `on_status` before running its delegate agent:

```python
@orchestrator_agent.tool
async def classify_request(ctx: RunContext[AppContext], customer_message: str) -> str:
    """Classify the customer's message into a category."""
    if ctx.deps.on_status:
        await ctx.deps.on_status("Classifying your request...")
    try:
        result = await classifier_agent.run(user_prompt=customer_message, usage=ctx.usage)
        return f"Category: {result.output.category.value}"
    except Exception as e:
        return f"ERROR: Classification failed: {str(e)}. Treat as general_query."
```

The `if ctx.deps.on_status:` guard is essential. Without it, calling `await None(...)` crashes. With it, tools silently skip status events when `on_status` is `None` (CLI mode, tests, non-streaming API).

#### Step 3: The generator creates a callback and drains the queue

In `triage_service.py`, the SSE generator creates an `asyncio.Queue` and a callback function that pushes SSE-formatted messages into the queue. The callback is stored in a new `AppContext` instance that gets passed to the orchestrator:

```python
async def run_triage_stream_events(ctx: AppContext, user_query: str) -> AsyncIterator[str]:
    status_queue: asyncio.Queue[str] = asyncio.Queue()

    async def emit_status(message: str):
        await status_queue.put(f"data: {json.dumps({'status': message})}\n\n")

    stream_ctx = AppContext(db=ctx.db, user_email=ctx.user_email, on_status=emit_status)

    async with orchestrator_agent.run_stream(user_prompt=user_query, deps=stream_ctx, ...) as result:
        # DRAIN: yield status events that arrived during tool calls
        while not status_queue.empty():
            yield await status_queue.get()

        async for partial_output in result.stream_output():
            # DRAIN: check for new status events between partial outputs
            while not status_queue.empty():
                yield await status_queue.get()
            if partial_output.customer_reply:
                yield f"data: {json.dumps({'customer_reply': partial_output.customer_reply})}\n\n"
```

#### Why the queue is needed

There's a timing challenge. The tool callbacks fire DURING `run_stream()` -- inside the `async with` block. But the `yield` for SSE can only happen at the generator level. The `asyncio.Queue` bridges this gap:

1. A tool calls `await ctx.deps.on_status("Classifying...")` -- which is `emit_status("Classifying...")`
2. `emit_status` puts the SSE string into `status_queue`
3. The generator drains `status_queue` at yield points and sends each message to the client

Without the queue, the callback would need to yield directly from inside a tool -- which is impossible in Python (only the generator function itself can yield).

**Critical: You must drain the queue.** If you create the queue and the callback but never read from the queue (`while not status_queue.empty(): yield await status_queue.get()`), the status messages pile up inside the queue and never reach the client. The drain loops must be placed at every yield point in the generator.

#### The full data flow

```
run_triage_stream_events() creates:
    - status_queue (asyncio.Queue)
    - emit_status function (puts SSE strings into queue)
    - stream_ctx with on_status=emit_status
        |
        v
orchestrator_agent.run_stream(deps=stream_ctx)
        |
        |-- calls classify_request tool
        |     ctx.deps.on_status("Classifying request...")
        |     → emit_status("Classifying request...")
        |     → status_queue.put("data: {"status": "Classifying request..."}\n\n")
        |
        |-- calls handle_support_request tool
        |     ctx.deps.on_status("Looking up your account...")
        |     → status_queue.put(...)
        |
        |-- starts streaming final output
              |
              v
        generator drains status_queue → yields all status events
        generator yields partial customer_reply events
        generator yields final response event
```

#### What the client sees

```
data: {"status": "Classifying your request..."}

data: {"status": "Looking up your account and order details..."}

data: {"status": "Escalating to a senior representative..."}

data: {"customer_reply": "We have received your refund request..."}

data: {"final": {"requires_human_approval": true, ...}}
```

#### The timing reality

Even with status events, there's a practical limitation. The orchestrator performs ALL tool calls before streaming its output. The status events are emitted during tool calls but only drained once `stream_output()` begins. So the status events arrive in a burst right when streaming starts, followed by partial outputs, followed by the final response. They're not truly progressive (arriving one by one during each tool call), but they still communicate what work was done -- which is valuable UX context.

---

## Multi-Agent Orchestration

There are two main patterns for running multiple agents together. This project evolved from the simpler pattern to the more powerful one.

### Pattern 1: Programmatic Hand-Off (simple, deterministic)

Python code calls agents in sequence. Your code decides the flow with `if/else` logic:

```python
async def run_triage(ctx: AppContext, user_query: str) -> FinalTriageResponse:
    # Python code calls classifier
    classifier_response = await classifier_agent.run(user_prompt=user_query)
    intent = classifier_response.output.category

    # Python code decides what to do next
    if intent == RequestCategory.GENERAL_QUERY:
        return FinalTriageResponse(...)

    # Python code calls specialist
    specialist_response = await specialist_agent.run(user_prompt=user_query, deps=ctx)
    return specialist_response.output
```

**Pros**: Deterministic flow, easy to debug, cheap (only calls agents when needed).
**Cons**: Rigid. Can't handle unexpected cases. Every routing decision must be hardcoded.

### Pattern 2: Agent Delegation (powerful, LLM-driven)

An orchestrator agent is the brain. It has tools that delegate to specialized agents. The LLM decides which agents to call and in what order:

```
Orchestrator Agent (the brain)
    |
    |-- tool: classify_request()        --> delegates to classifier_agent
    |-- tool: handle_support_request()  --> delegates to specialist_agent
    |-- tool: escalate_to_manager()     --> delegates to escalation_agent
    |
    Orchestrator sees all results, reasons about them, returns final output
```

**Pros**: Flexible routing, handles unexpected cases, can call tools in any order or combination.
**Cons**: Higher token cost (multiple agent round-trips), non-deterministic tool call order, harder to debug.

### How Agent Delegation Works

The orchestrator agent has tools. Each tool internally runs a delegate agent and returns the result as a serialized string:

```python
orchestrator_agent = Agent(
    model="openai:gpt-4.1-mini",
    output_type=FinalTriageResponse,
    deps_type=AppContext,
    retries=3
)

@orchestrator_agent.tool
async def classify_request(ctx: RunContext[AppContext], customer_message: str) -> str:
    """Classify the customer's message into a category."""
    result = await classifier_agent.run(
        user_prompt=customer_message,
        usage=ctx.usage,      # Roll up token usage to the parent
    )
    return f"Category: {result.output.category.value}"

@orchestrator_agent.tool
async def handle_support_request(ctx: RunContext[AppContext], customer_message: str) -> str:
    """Handle a complex customer support request with database lookups."""
    result = await specialist_agent.run(
        user_prompt=customer_message,
        deps=ctx.deps,         # Pass dependencies to the delegate
        usage=ctx.usage,       # Roll up token usage to the parent
    )
    return result.output.model_dump_json()

@orchestrator_agent.tool
async def escalate_to_manager(ctx: RunContext[AppContext], customer_message: str, reason: str) -> str:
    """Escalate a high-risk case to a human manager."""
    result = await escalation_agent.run(
        user_prompt=f"Customer message: {customer_message}\nEscalation reason: {reason}",
        deps=ctx.deps,
        usage=ctx.usage,
    )
    return result.output.model_dump_json()
```

### Critical Rules for Agent Delegation

**Rule 1: Always pass `usage=ctx.usage` to delegate agents.**

This rolls all token usage from every delegate agent up to the parent. When you call `orchestrator_result.usage()`, you get the TOTAL cost across ALL agents in the chain -- orchestrator + classifier + specialist + escalation. Without this, you lose visibility into the true cost of a run.

**Rule 2: Pass `deps=ctx.deps` to delegates that need dependencies.**

The orchestrator's `AppContext` (database, user email) must be forwarded to delegates. The specialist needs DB access for its tools. The escalation agent needs the user email for its system prompt. Use `ctx.deps` to pass them through.

**Rule 3: Delegate tools return strings, not Pydantic models.**

The delegate agent returns structured output internally (e.g., `FinalTriageResponse`). But the tool function must serialize it to a string for the orchestrator using `model_dump_json()`. Why? Because tools communicate with the orchestrator LLM via text. The orchestrator is an LLM -- it can only read text. When a tool returns a value, that value becomes part of the conversation that the orchestrator reads. If you return a Python object, the LLM receives something like `<FinalTriageResponse object at 0x...>` which is useless. If you return `model_dump_json()`, the LLM receives readable JSON like `{"customer_reply": "Your refund is being processed...", "order_id": "#123", ...}` that it can reason about and incorporate into its own final response.

**Rule 4: Add try/except inside delegate tools.**

If a delegate agent fails (connection error, validation error, token limit exceeded), the exception propagates and crashes the entire orchestrator run. Catch errors inside the tool and return an error string so the orchestrator can handle it gracefully:

```python
@orchestrator_agent.tool
async def handle_support_request(ctx: RunContext[AppContext], customer_message: str) -> str:
    """Handle a complex customer support request."""
    try:
        result = await specialist_agent.run(
            user_prompt=customer_message,
            deps=ctx.deps,
            usage=ctx.usage,
        )
        return result.output.model_dump_json()
    except Exception as e:
        return f"ERROR: Specialist agent failed: {str(e)}. Respond to the customer directly."
```

**Error message design matters.** The error string goes straight to the orchestrator LLM as a tool result. The orchestrator reads it and decides what to do next. Each delegate tool should have a different fallback instruction tailored to that tool's role:

```python
# Classifier fails: suggest a safe default category
return f"ERROR: Classification failed: {str(e)}. Treat as general_query."

# Specialist fails: tell orchestrator to compose its own reply
return f"ERROR: Specialist agent failed: {str(e)}. Respond to the customer directly."

# Escalation fails: ensure the case still gets flagged
return f"ERROR: Escalation failed: {str(e)}. Flag for manual review."
```

The orchestrator LLM reads these instructions and adapts. If the classifier fails, it treats the request as a general query. If the specialist fails, it writes a response itself. If escalation fails, it flags the case for human review. Each error message is a mini-instruction for graceful degradation.

**Rule 5: The orchestrator's system prompt guides but doesn't guarantee tool order.**

You can instruct the orchestrator to "ALWAYS classify first, then handle support." But the LLM might ignore this. It might skip classification and go straight to the specialist. It might call escalation without calling the specialist first. You cannot guarantee tool call order the same way you can with Python `if/else`. This is the fundamental trade-off of letting the LLM be the brain.

### Deep Dive: `deps` vs `usage` -- When to Pass Which and Why

When calling a delegate agent from inside an orchestrator tool, you have two context parameters to consider: `deps` and `usage`. They serve completely different purposes, and not every delegate needs both.

| Parameter | What it does | When to pass it |
|---|---|---|
| `deps=ctx.deps` | Forwards the `AppContext` (database, user email) to the delegate | Only if the delegate's tools or system prompt need dependency access |
| `usage=ctx.usage` | Rolls the delegate's token consumption up to the parent's usage tracker | **Always** -- every delegate should report its cost to the parent |

**Example from our project:**

```python
# classify_request: passes usage ONLY
# Why: The classifier has no tools, no database access, no system prompt that needs user email.
# It just reads the message and returns a category. It doesn't need AppContext.
result = await classifier_agent.run(
    user_prompt=customer_message,
    usage=ctx.usage,       # YES: track cost
    # deps NOT passed      # The classifier doesn't need AppContext
)

# handle_support_request: passes BOTH
# Why: The specialist has tools (fetch_user_tier, fetch_order_status) that call ctx.deps.db
# and a system prompt that reads ctx.deps.user_email. It needs the full AppContext.
result = await specialist_agent.run(
    user_prompt=customer_message,
    deps=ctx.deps,         # YES: specialist needs DB and user email
    usage=ctx.usage,       # YES: track cost
)

# escalate_to_manager: passes BOTH
# Why: The escalation agent's system prompt reads ctx.deps.user_email
result = await escalation_agent.run(
    user_prompt=...,
    deps=ctx.deps,         # YES: escalation needs user email for its prompt
    usage=ctx.usage,       # YES: track cost
)
```

**What happens if you forget `usage=ctx.usage`?** The delegate agents still run fine. But when you call `orchestrator_result.usage()`, you only see the orchestrator's own token consumption. The classifier's 300 tokens, the specialist's 2000 tokens, and the escalation's 800 tokens are invisible. You think the run cost 1000 tokens when it actually cost 4100. In production, this means your cost monitoring is wrong and your budget estimates are off.

**What happens if you forget `deps=ctx.deps`?** The delegate agent crashes immediately if any of its tools or system prompts try to access `ctx.deps`. You'll get an error like `AttributeError: 'NoneType' object has no attribute 'db'` inside the tool function.

### Deep Dive: Exception Propagation Through Agent Layers

In a delegation pattern, exceptions can propagate through multiple layers. Understanding the path is critical for debugging:

```
Layer 1: main.py calls run_triage()
    Layer 2: run_triage() calls orchestrator_agent.run()
        Layer 3: orchestrator calls classify_request tool
            Layer 4: classify_request calls classifier_agent.run()
                --> classifier_agent connects to Ollama
                --> Ollama is not running
                --> ConnectionError raised

Without try/except in the tool:
    ConnectionError propagates from Layer 4 --> Layer 3 --> Layer 2
    orchestrator_agent.run() crashes
    run_triage()'s except block catches it
    Returns fallback FinalTriageResponse("I'm sorry, I'm unable to process your request.")

With try/except in the tool:
    ConnectionError caught at Layer 3 (inside classify_request tool)
    Tool returns: "ERROR: Classification failed: Connection error"
    Orchestrator LLM reads the error string
    Orchestrator can decide: try again, skip classification, or report the issue in its response
    Much more graceful degradation
```

**The same applies to validation errors, token limit exceeded, and network timeouts.** Any unhandled exception inside a delegate tool kills the entire orchestrator run. Always wrap delegate calls in `try/except` inside the tool function and return a descriptive error string.

**Debugging tip:** When something goes wrong in a 3-layer agent chain, the top-level error message is often generic ("Orchestrator failed: Connection error"). To find the real cause, add logging inside each delegate tool:

```python
@orchestrator_agent.tool
async def classify_request(ctx: RunContext[AppContext], customer_message: str) -> str:
    try:
        result = await classifier_agent.run(user_prompt=customer_message, usage=ctx.usage)
        print(f"[DEBUG] Classifier result: {result.output.category.value}")
        return f"Category: {result.output.category.value}"
    except Exception as e:
        print(f"[ERROR] Classifier failed: {type(e).__name__}: {e}")
        return f"ERROR: Classification failed: {str(e)}"
```

### How the Triage Service Simplifies with Delegation

Before (programmatic hand-off): the triage service contained all routing logic with `if/else`:

```python
# Before: Python code orchestrates everything
classifier_response = await classifier_agent.run(...)
intent = classifier_response.output.category
if intent == RequestCategory.GENERAL_QUERY:
    return FinalTriageResponse(...)
specialist_response = await specialist_agent.run(...)
# ... many more lines of routing logic
```

After (agent delegation): the triage service just calls the orchestrator and applies business rules:

```python
# After: Orchestrator agent handles all routing
result = await orchestrator_agent.run(user_prompt=user_query, deps=ctx, usage_limits=...)
output = apply_business_rules(ctx, result.output)
return output
```

The orchestrator decides which agents to call. Your Python code only enforces business rules on the final output. Much simpler.

### Token Cost Implications of Delegation

Agent delegation significantly increases token usage. Every delegate call is a full agent run with its own round-trips:

```
Orchestrator request 1: system prompt + user query + tool definitions  (~500 tokens)
Orchestrator request 2: decides to call classify_request tool
    --> Classifier run: ~300 tokens
Orchestrator request 3: receives classification, decides to call specialist
    --> Specialist run: ~2000 tokens (includes its own tool calls)
Orchestrator request 4: receives specialist result, decides to call escalation
    --> Escalation run: ~800 tokens
Orchestrator request 5: composes final FinalTriageResponse
```

Total: 5+ orchestrator requests + 3 delegate agent runs = potentially 5000-10000 tokens per customer query. Set `total_tokens_limit` accordingly (12000-15000 for delegation patterns).

Monitor with `result.usage()` -- this shows the aggregate across ALL agents when you use `usage=ctx.usage`.

### Running Triage in Parallel

Use `asyncio.gather` to run independent triage calls concurrently:

```python
results = await asyncio.gather(
    run_triage(AppContext(db=db, user_email="user1@example.com"), "I want a refund"),
    run_triage(AppContext(db=db, user_email="user2@example.com"), "What are your hours?"),
)
```

Each call gets its own `AppContext` with the specific user's email, but all share the same database instance.

---

## Model Selection Guidelines

| Task Type | Recommended Model | Why |
|---|---|---|
| Intent classification | Local model (`ollama:llama3.2`) | Cheap, fast, no API cost. Simple categorization. |
| Structured output with tool use | `openai:gpt-4.1-mini` or `openai:gpt-4o-mini` | Good at following schemas, no reasoning overhead. |
| Complex multi-step reasoning | `openai:gpt-5-nano` or `openai:gpt-4o` | Only when the task genuinely requires deep reasoning. |

Always verify your model name is valid. Use `result.usage()` to check for unexpected reasoning token overhead.

---

## Common Mistakes to Avoid

### 1. Not passing runtime data through dependencies
**Wrong**: Hoping the LLM will figure out the user's email from the conversation.
**Right**: Put `user_email` in `AppContext`, inject it via `@agent.system_prompt`.

### 2. Hardcoding values that should come from agent output
**Wrong**: `process_refund(order_id="#123")` -- hardcoded order ID.
**Right**: Add `order_id` to the output schema, use `result.output.order_id`.

### 3. Trusting the LLM for business decisions
**Wrong**: Relying on the LLM to set `requires_human_approval=True` for refunds.
**Right**: Check the classifier's intent in Python and override the flag.

### 4. Manual retry loops instead of ModelRetry
**Wrong**: `for attempt in range(3): try: ... except: message_history.append(fake_message)`
**Right**: Set `retries=3` on the agent, use `@agent.output_validator` with `ModelRetry`.

### 5. Adding LLM-verifiable fields to the schema
**Wrong**: Adding `is_order_found: bool` to the schema -- the LLM might get it wrong.
**Right**: Check `ctx.db.get_order_status(order_id)` in your Python code.

### 6. Catching all exceptions from agent runs
**Wrong**: `except Exception` that swallows network errors, auth failures, and validation errors alike.
**Right**: Let infrastructure errors propagate. Only handle expected agent failures gracefully.

### 7. No usage limits
**Wrong**: Running an agent with no cost controls -- a tool-call loop can burn your entire budget.
**Right**: Always set `UsageLimits(request_limit=..., total_tokens_limit=...)`.

### 8. Output validator without partial_output guard
**Wrong**: Validator checks field lengths without considering streaming partial outputs. The validator fires on empty fields before the model finishes generating them, killing the stream.
**Right**: Always add `if ctx.partial_output: return output` as the first line of every output validator.

### 9. Using `result.output` with streaming
**Wrong**: `output = result.output` after `run_stream()` -- `StreamedRunResult` has no `.output` attribute.
**Right**: `output = await result.get_output()` -- must explicitly await the final output.

### 10. Duplicating business rules across run modes
**Wrong**: Copy-pasting business rule logic into both `run_triage()` and `run_triage_streaming()`.
**Right**: Extract into `apply_business_rules()` and call it from both functions.

### 11. Using reasoning models for simple tasks
**Wrong**: Using `gpt-5-nano` (reasoning model) for a customer support reply. Burns 1700+ reasoning tokens on internal chain-of-thought for a one-sentence answer.
**Right**: Use non-reasoning models like `gpt-4.1-mini` for straightforward tasks. Reserve reasoning models for tasks that genuinely require multi-step thinking. Check `result.usage()` for unexpected reasoning token overhead.

### 12. Not passing `usage=ctx.usage` to delegate agents
**Wrong**: Calling a delegate agent without `usage=ctx.usage`. The parent agent has no visibility into the delegate's token cost. `orchestrator_result.usage()` only shows the orchestrator's own tokens, hiding the real cost.
**Right**: Always pass `usage=ctx.usage` to every delegate agent call so usage rolls up to the parent.

### 13. Letting delegate exceptions crash the orchestrator
**Wrong**: A delegate tool function that lets exceptions propagate. If the specialist agent fails (network error, validation error), the entire orchestrator run crashes.
**Right**: Wrap delegate calls in `try/except` inside the tool function. Return an error string so the orchestrator can handle it gracefully.

### 14. Returning Pydantic models from delegate tools
**Wrong**: `return result.output` from a delegate tool -- the orchestrator receives a Python object it cannot interpret.
**Right**: `return result.output.model_dump_json()` -- serialize to JSON string so the orchestrator can read and reason about the data.

### 15. Missing fields in manual FinalTriageResponse constructors
**Wrong**: Creating fallback `FinalTriageResponse(requires_human_approval=True, order_id=None, ...)` without all required fields. If you add a new field like `category` to the schema, every manual constructor in fallback/exception paths will fail with `ValidationError: category Field required`.
**Right**: When adding new required fields to output schemas, search for every manual constructor of that model (especially in `except` blocks) and add the new field with a sensible default like `category="unknown"`.

### 16. Using `print()` instead of `logging`
**Wrong**: Scattered `print()` calls throughout service code for debugging and operational events.
**Right**: Use `logging` with severity levels (`INFO`, `WARNING`, `ERROR`). Create a shared `get_logger(__name__)` factory in `src/logger.py`. Keep `print()` only for intentional CLI output (user-facing text in `main.py`).

### 17. Putting structured data in `extra={}` with a standard formatter
**Wrong**: `logger.info("Run complete", extra={"tokens": 222})` -- the `extra` dict is attached to the log record but invisible in the output because the standard format string `%(asctime)s - %(name)s - %(levelname)s - %(message)s` doesn't reference custom field names.
**Right**: Embed the data directly in the message: `logger.info(f"Run complete | tokens={usage.output_tokens} | category={output.category.value}")`.

### 18. Using ModelRetry when the LLM will re-extract the same data on every retry
**Wrong**: Raising `ModelRetry` in a validator when the rejection is triggered by a field the LLM correctly extracted from the user's message. Example: user says "refund order #99990", LLM sets `order_id="#99990"`, validator checks DB and rejects because the order doesn't exist, ModelRetry tells LLM to set `order_id` to null. The LLM ignores this -- the user mentioned the order, the schema says to include it, and the schema wins. Every retry fails, retries are exhausted, the fallback fires.
**Right**: Use a deterministic override in `apply_business_rules`. When Python checks the DB and the order doesn't exist, return a new `FinalTriageResponse` with the correct `customer_reply` directly. No LLM negotiation needed.

### 19. Business rules that blindly overwrite each other (last-writer-wins bug)
**Wrong**: Sequential flag assignments where a later rule can silently undo an earlier one. Example: Rule 1 sets `requires_human_approval = True` for refunds. Rule 2 sets `requires_human_approval = False` for non-existent orders. For a refund on a non-existent order, Rule 2 undoes Rule 1, and the refund gets no human review.
**Right**: Make rules category-aware, or use an early return when a rule produces a complete answer. If an order doesn't exist, return the entire override `FinalTriageResponse` immediately rather than just setting a flag that a subsequent rule might overwrite.

### 21. Creating a callback queue but never draining it
**Wrong**: Creating an `asyncio.Queue` and an `emit_status` callback, storing it in `AppContext`, but never reading from the queue in the generator. Status messages pile up in the queue and never reach the client.
**Right**: Add `while not status_queue.empty(): yield await status_queue.get()` at every yield point in the generator -- both before and inside the `async for partial_output` loop.

### 22. Calling `model_dump` without parentheses
**Wrong**: `error_response.model_dump` -- references the method object itself, which serializes to something like `<bound method BaseModel.model_dump>`.
**Right**: `error_response.model_dump()` -- calls the method and returns the actual dictionary.

---

## Testing Agent Systems

Testing AI agent systems requires a different mindset than testing traditional software. You can't predict exactly what an LLM will say, but you CAN test everything around it: business rules, validators, data transformations, and error handling. These deterministic components are where bugs hide and where tests provide the most value.

### The testing principle for AI systems

Separate what's deterministic from what's probabilistic:

| Component | Deterministic? | Testable without LLM? | Test approach |
|---|---|---|---|
| `apply_business_rules()` | Yes | Yes | Construct fake outputs, verify rules fire correctly |
| Output validators | Yes | Yes | Construct fake outputs, verify `ModelRetry` is raised or not |
| Tool functions | Yes | Yes | Call directly with fake context, verify return values |
| LLM response quality | No | No | Manual testing, eval frameworks, monitoring |

Focus your automated tests on the deterministic layer. This is where you get fast, free, reliable tests that run in milliseconds with zero API cost.

### Setup: Install pytest

```bash
uv add --dev pytest pytest-asyncio
```

`--dev` marks these as development dependencies -- needed for testing, not for running the app.

### Test file structure

```
tests/
├── test_business_rules.py    # Tests for apply_business_rules()
├── test_validators.py        # Tests for output validators
```

### Testing business rules

Business rule tests verify that `apply_business_rules()` correctly overrides LLM output. The pattern: construct a `FinalTriageResponse` manually (simulating what the LLM would return), pass it through the business rules, and assert the result.

```python
import pytest
from src.config import AppContext
from src.db import MockDB
from src.schemas import FinalTriageResponse, RequestCategory
from src.triage_service import apply_business_rules


@pytest.fixture
def db():
    """Fresh MockDB instance for each test"""
    return MockDB()


@pytest.fixture
def ctx(db):
    """AppContext with a test user"""
    return AppContext(db=db, user_email="user1@gmail.com")
```

**Fixtures** are reusable setup functions. Pytest automatically injects them into any test that lists them as parameters. Each test gets a fresh instance -- test A can't corrupt data for test B.

**Fixture chains**: The `ctx` fixture depends on `db`. Pytest sees this, runs `db()` first, passes the result into `ctx(db)`. You don't manage the dependency order -- pytest does it for you.

#### Testing that a business rule fires

```python
def test_refund_forces_human_approval(ctx):
    """Even if the LLM says no approval needed, refunds always require it."""
    output = FinalTriageResponse(
        requires_human_approval=False,  # LLM says no
        order_id="#123",
        category=RequestCategory.REFUND,
        suggested_action="Process refund",
        customer_reply="Your refund is being processed."
    )
    result = apply_business_rules(ctx, output)
    assert result.requires_human_approval is True
```

What's happening:
1. We construct a `FinalTriageResponse` with `requires_human_approval=False` -- simulating an LLM that got it wrong
2. We pass it through `apply_business_rules` -- the same function that runs in production
3. We assert the result is `True` -- the refund rule must override the LLM's decision

No LLM was called. We manually constructed the output object to simulate a specific scenario.

#### Testing that a business rule does NOT fire

```python
def test_tech_support_with_existing_order_preserves_llm_decision(ctx):
    """For non-refund categories with a valid order, the LLM's decision stands."""
    output = FinalTriageResponse(
        requires_human_approval=False,
        order_id="#123",
        category=RequestCategory.TECHNICAL_SUPPORT,
        suggested_action="Help with technical issue",
        customer_reply="Let me help you with that."
    )
    result = apply_business_rules(ctx, output)
    assert result.requires_human_approval is False
```

This proves that when no business rule applies, the LLM's judgment passes through unchanged. Equally important as testing that rules fire -- you need to verify they stay out of the way when they shouldn't intervene.

#### Testing data integrity

```python
def test_other_fields_preserved(ctx):
    """Business rules should only change requires_human_approval, not other fields."""
    output = FinalTriageResponse(
        requires_human_approval=False,
        order_id="#123",
        category=RequestCategory.REFUND,
        suggested_action="Process the refund immediately",
        customer_reply="Your refund for order #123 is confirmed."
    )
    result = apply_business_rules(ctx, output)

    assert result.requires_human_approval is True     # Changed by rule
    assert result.order_id == "#123"                   # Preserved
    assert result.category == RequestCategory.REFUND   # Preserved
    assert result.suggested_action == "Process the refund immediately"  # Preserved
    assert result.customer_reply == "Your refund for order #123 is confirmed."  # Preserved
```

When a function transforms data, test both **what changed** and **what should NOT have changed**. The "should not have changed" tests catch the sneakiest bugs -- someone refactors `apply_business_rules`, accidentally forgets to pass `customer_reply` through, and this test catches it instantly.

#### What to test for business rules

Test every combination of conditions that affect the output:

| Scenario | Expected `requires_human_approval` | Why |
|---|---|---|
| Refund + LLM says False | True | Refund rule overrides |
| Refund + LLM says True | True | Refund rule agrees, no change |
| Refund + non-existent order (#999) | False | Order doesn't exist, nothing to approve |
| Refund + no order (None) | False | No order, nothing to approve |
| General query + no order | False | No rule fires, LLM's decision stands |
| General query + LLM says True + no order | False | No order rule overrides |
| Tech support + existing order | (LLM's value) | No rule fires |
| Tech support + non-existent order | False | Order doesn't exist rule fires |

### Testing output validators

Output validators are also deterministic functions. The challenge: they take `RunContext` as a parameter, which Pydantic AI normally creates internally. Solution: use `MagicMock` to fake it.

```python
from unittest.mock import MagicMock
from pydantic_ai import ModelRetry
from src.agents import validate_specialist_output, validate_escalation, validate_orchestrator


@pytest.fixture
def mock_ctx():
    """Fake RunContext for testing validators outside of a real agent run."""
    ctx = MagicMock()
    ctx.partial_output = False  # Test final output validation by default
    return ctx
```

**`MagicMock`**: A fake object from Python's standard library that pretends to be anything. When you access any attribute, it returns another `MagicMock` instead of crashing. We only need to set `ctx.partial_output = False` -- the validator checks this first, and we want to test the real validation logic (not the streaming skip).

#### Testing that good output passes through

```python
def test_specialist_valid_output_passes(mock_ctx):
    """Good output should pass through the validator unchanged."""
    output = FinalTriageResponse(
        requires_human_approval=False,
        order_id="#123",
        category=RequestCategory.REFUND,
        suggested_action="Process the refund for order #123",
        customer_reply="Your refund for order #123 is being processed. You'll receive confirmation within 3-5 business days."
    )
    result = validate_specialist_output(mock_ctx, output)
    assert result == output
```

Pydantic `BaseModel` objects support equality comparison -- two instances are equal if all their field values match. So `result == output` verifies every field came through unchanged.

#### Testing that bad output triggers ModelRetry

```python
def test_specialist_rejects_short_customer_reply(mock_ctx):
    """customer_reply shorter than 10 characters should trigger ModelRetry."""
    output = FinalTriageResponse(
        requires_human_approval=False,
        order_id="#123",
        category=RequestCategory.REFUND,
        suggested_action="Process the refund",
        customer_reply="Ok."
    )
    with pytest.raises(ModelRetry):
        validate_specialist_output(mock_ctx, output)
```

**`pytest.raises(ModelRetry)`**: This block says "I expect this code to raise a `ModelRetry` exception. If it does, the test passes. If it doesn't raise, the test fails." It's the opposite of a normal assertion -- you're verifying that the function rejects the input.

**Test one validation rule at a time.** If the validator checks both `customer_reply` length and `suggested_action` length, write separate tests for each. One test has a short `customer_reply` but a good `suggested_action`. The other has a good `customer_reply` but a short `suggested_action`. If a test fails, you know exactly which rule broke.

#### Testing the streaming guard

```python
def test_specialist_skips_validation_for_partial_output(mock_ctx):
    """During streaming, partial outputs should skip validation entirely."""
    mock_ctx.partial_output = True  # Override fixture default

    output = FinalTriageResponse(
        requires_human_approval=False,
        order_id=None,
        category=RequestCategory.GENERAL_QUERY,
        suggested_action="",       # Would normally fail validation
        customer_reply=""           # Would normally fail validation
    )
    result = validate_specialist_output(mock_ctx, output)
    assert result == output
```

Both `suggested_action` and `customer_reply` are empty -- both would trigger `ModelRetry` in normal validation. But `ctx.partial_output = True` simulates streaming, so the validator skips all checks and returns the output immediately. This test guarantees the streaming guard works and can never be accidentally removed.

### Running tests

```bash
# Run a specific test file
python -m pytest tests/test_business_rules.py -v

# Run all tests in the tests/ directory
python -m pytest tests/ -v

# Run with -v for verbose output (shows each test name and PASSED/FAILED)
```

Use `python -m pytest` (not just `pytest`) to ensure Python sets up import paths correctly from your project root.

### Testing principles for agent systems

1. **Test deterministic code, not LLM output.** You can't reliably assert what an LLM will say. You CAN reliably assert that your business rules, validators, and data transformations work correctly for every possible LLM output.

2. **Construct fake outputs manually.** Don't call the LLM in tests. Build `FinalTriageResponse` objects directly with the specific field values you want to test. This is fast, free, and deterministic.

3. **Test one rule per test function.** Each test should verify one specific behavior. If a test fails, the name tells you exactly what broke: `test_refund_forces_human_approval` is immediately clear.

4. **Test both sides of every rule.** For each business rule, test that it fires when it should AND that it doesn't fire when it shouldn't. Missing the "doesn't fire" test means you might not catch a rule that's too aggressive.

5. **Test edge cases at boundaries.** If a validator rejects strings shorter than 10 characters, test with 3 characters (clearly short), 9 characters (just under), and 10+ characters (passes). Boundary bugs are common.

6. **Use fixtures for shared setup.** Database instances, contexts, and mock objects should be fixtures, not repeated setup code in every test function.

7. **Use `MagicMock` for framework objects.** When testing functions that take Pydantic AI's `RunContext`, create a fake with `MagicMock()` and set only the attributes your function actually accesses.

---

## API Layer with FastAPI

Wrapping your agent system in an HTTP API transforms it from a CLI script into a real service. FastAPI is the natural choice because it's built on Pydantic -- the same models you use for agent output serve as API response schemas with zero conversion.

### Why FastAPI + Pydantic AI work together

- `FinalTriageResponse` is both the agent's `output_type` and the API's `response_model`
- `TriageRequest` validates incoming HTTP requests the same way Pydantic validates agent output
- Both are async-native -- FastAPI's `async def` endpoints and Pydantic AI's `await agent.run()` use the same event loop
- FastAPI auto-generates interactive API docs from your Pydantic models at `/docs`

### Setup

```bash
uv add fastapi uvicorn[standard]
```

- `fastapi` is the web framework
- `uvicorn` is the ASGI server. ASGI (Asynchronous Server Gateway Interface) is the async version of WSGI -- it natively supports `async/await`, which is what your agent code needs

### Request schema

Add an input model to `schemas.py` alongside your existing output schemas:

```python
class TriageRequest(BaseModel):
    """The API request input to the triage agent"""
    email: str
    query: str
```

FastAPI validates incoming requests against this model automatically. If someone sends a request without `email` or `query`, FastAPI returns a 422 error before your code even runs.

### The API application

```python
from fastapi import FastAPI, HTTPException
from src.db import MockDB
from src.config import AppContext
from src.schemas import TriageRequest, FinalTriageResponse
from src.triage_service import run_triage

app = FastAPI(
    title="AI Customer Support API",
    description="Multi-agent customer support system powered by Pydantic AI"
)

# Shared database -- created once at server startup, used by all requests
db_instance = MockDB()

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/triage", response_model=FinalTriageResponse)
async def triage(request: TriageRequest):
    ctx = AppContext(db=db_instance, user_email=request.email)
    result = await run_triage(ctx, request.query)

    # Detect orchestrator failure (fallback returns category="unknown")
    if result.category == "unknown":
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Service temporarily unavailable. Please try again later.",
                "message": result.customer_reply,
                "suggested_action": result.suggested_action,
            }
        )

    return result
```

### How async concurrency works (no thread blocking)

This is the most important concept for the API layer. Your agent calls are `async` -- they use `await` for network I/O (LLM API calls). While one request waits for the LLM to respond, the event loop is free to handle other requests.

```
Request 1 arrives: POST /triage {"email": "user1@...", "query": "I want a refund"}
    --> FastAPI creates async task 1
    --> task 1 calls await run_triage()
    --> orchestrator makes LLM API call (network I/O)
    --> WHILE WAITING: event loop is free

Request 2 arrives: POST /triage {"email": "user2@...", "query": "Business hours?"}
    --> FastAPI creates async task 2 (task 1 is still waiting)
    --> task 2 calls await run_triage()
    --> task 2's LLM call starts in parallel

Request 1's LLM responds --> task 1 resumes, applies business rules, returns 200
Request 2's LLM responds --> task 2 resumes, applies business rules, returns 200
```

**Key difference from `main.py`**: In `main.py`, you use `asyncio.gather()` to run multiple requests in parallel. In FastAPI, parallelism is automatic. Each HTTP request is its own async task. The ASGI server handles concurrency for you -- every `await` is a yield point where other requests can be processed.

**No existing code changes needed.** The API calls `run_triage()` -- the same function that `main.py` calls. Your agents, business rules, validators, and schemas are completely untouched. The API is a new layer on top.

### Error handling with HTTP status codes

Without proper error handling, the API returns HTTP 200 even when the orchestrator fails (because `run_triage` catches exceptions and returns a fallback `FinalTriageResponse`). The caller has no way to distinguish success from failure by status code alone.

The solution: detect the fallback output and return an appropriate HTTP error.

```python
# Detect orchestrator failure
if result.category == "unknown":
    raise HTTPException(
        status_code=503,
        detail={
            "error": "Service temporarily unavailable. Please try again later.",
            "message": result.customer_reply,
            "suggested_action": result.suggested_action,
        }
    )
```

**Why `category == "unknown"` is the signal:** When the orchestrator fails, the `except` block in `run_triage` returns a fallback `FinalTriageResponse` with `category="unknown"`. This is the only time `"unknown"` appears -- successful runs always produce a real `RequestCategory` value. The API layer checks for this and converts it to an HTTP 503.

**Status code choice:**
- `200` -- Success. The agent processed the request normally.
- `422` -- Validation error. FastAPI returns this automatically when the request body doesn't match `TriageRequest`.
- `503` -- Service Unavailable. The orchestrator failed (LLM connection error, token limit exceeded, model unavailable). Signals a temporary issue -- the client should retry.

### Layered error handling architecture

With the API layer, errors are handled at three levels:

```
Layer 1: Delegate tool try/except (agents.py)
    --> Catches classifier/specialist/escalation failures
    --> Returns error string to orchestrator
    --> Orchestrator continues with degraded info (graceful degradation)

Layer 2: run_triage except (triage_service.py)
    --> Catches orchestrator-level failures (OpenAI down, token limit)
    --> Returns fallback FinalTriageResponse with category="unknown"

Layer 3: API HTTPException check (api.py)
    --> Detects fallback by checking category == "unknown"
    --> Returns HTTP 503 with detailed error JSON
```

Each layer handles a different scope of failure. Layer 1 absorbs delegate failures so the orchestrator can continue. Layer 2 absorbs orchestrator failures so the API doesn't crash. Layer 3 translates the fallback into a proper HTTP error so the client knows what happened.

**Important insight:** If only a delegate fails (e.g., Ollama is down for the classifier), Layer 1 handles it. The orchestrator adapts, produces a valid response, and the API returns HTTP 200. The client never knows a delegate failed -- the system degraded gracefully. Only when the orchestrator itself fails does the client see a 503.

### Streaming endpoint with Server-Sent Events (SSE)

The standard `/triage` endpoint returns one JSON response when the agent finishes. The streaming `/triage/stream` endpoint sends progressive updates as the agent works, using the Server-Sent Events protocol.

**SSE format**: Each event is a line starting with `data: ` followed by a JSON payload, followed by two newlines (`\n\n`). This is the standard SSE text format that browsers and HTTP clients understand.

```python
from fastapi.responses import StreamingResponse
from src.triage_service import run_triage_stream_events

@app.post("/triage/stream")
async def triage_stream(request: TriageRequest):
    ctx = AppContext(db=db_instance, user_email=request.email)
    return StreamingResponse(
        run_triage_stream_events(ctx, request.query),
        media_type="text/event-stream"
    )
```

**How it works:**
1. FastAPI receives the POST request and creates an `AppContext`
2. `StreamingResponse` wraps the `run_triage_stream_events` async generator
3. Every time the generator `yield`s an SSE string, FastAPI sends it immediately to the client
4. The connection stays open until the generator finishes (yields the final event)

The generator yields three types of events:
- `{"status": "Classifying your request..."}` -- tool call progress (from the callback/queue pattern)
- `{"customer_reply": "We have received..."}` -- partial structured output as tokens arrive
- `{"final": {...}}` -- the complete `FinalTriageResponse` with business rules applied

**Testing with curl:**

```bash
# -N disables curl's output buffering so you see events as they arrive
curl -X POST http://localhost:8000/triage/stream \
    -H "Content-Type: application/json" \
    -d '{"email": "user1@gmail.com", "query": "I want a refund for order #123"}' \
    -N
```

**Note:** Swagger UI (`/docs`) doesn't handle SSE streaming properly -- it waits for the full response and shows everything at once. Use `curl -N` to see events arrive progressively.

### Running the API

```bash
# Start the server with auto-reload for development
uvicorn api:app --reload

# Test standard endpoint
curl -X POST http://localhost:8000/triage \
    -H "Content-Type: application/json" \
    -d '{"email": "user1@gmail.com", "query": "I want a refund for order #123"}'

# Test streaming endpoint
curl -X POST http://localhost:8000/triage/stream \
    -H "Content-Type: application/json" \
    -d '{"email": "user1@gmail.com", "query": "I want a refund for order #123"}' \
    -N

# Interactive docs (auto-generated from Pydantic models)
# Open in browser: http://localhost:8000/docs
```

---

## Interactive UI with NiceGUI

The project has three entry points that all share the same core service layer:

| Entry point | Purpose | How to run |
|---|---|---|
| `main.py` | CLI batch testing, human-in-the-loop approval | `python main.py` |
| `api.py` | HTTP API (standard + streaming endpoints) | `uvicorn api:app --reload` |
| `ui.py` | Interactive browser UI | `python ui.py` |

All three call the same `run_triage_stream_events` generator from `triage_service.py`. No business logic is duplicated.

### Why NiceGUI

NiceGUI is a Python-first web UI framework. You write the entire UI in Python -- no HTML, no JavaScript, no separate frontend build step. It runs a FastAPI server internally and communicates with the browser via WebSocket. This makes it the natural choice for wrapping a Python AI agent in an interactive demo.

Key fit with this project:
- Natively async -- NiceGUI's `async def` event handlers work directly with `await` and `async for`, so consuming the async generator from `run_triage_stream_events` requires zero adaptation
- No frontend build -- the entire UI is `ui.py`, one file
- Component library covers everything needed: inputs, cards, badges, logs, progress indicators

### Install

```bash
uv add nicegui
```

### UI architecture

The page has five visible areas, all hidden by default and revealed progressively as events arrive:

```
[Header]

[Input Card]
    Email input | Query textarea    (side by side, same height)
    [Run Agent] button

[Status Label]                      ← updates from "status" tool events only
                                      e.g. "⚙ Classifying your request..."

[Customer Reply Card]               ← hidden until "final" event arrives
    Full reply text

[Triage Result Card]                ← hidden until "final" event arrives
    Category badge (color-coded)
    Human Approval badge
    Order ID
    Suggested Action

[Raw Event Log]                     ← collapsible, shows raw SSE strings
```

### Parsing SSE events in the UI

The `run_triage_stream_events` generator yields raw SSE-formatted strings. The UI strips the `data: ` prefix and parses the JSON:

```python
def parse_sse_event(raw: str) -> dict | None:
    """Strip the SSE 'data: ' prefix and parse the JSON payload."""
    line = raw.strip()
    if line.startswith("data: "):
        try:
            return json.loads(line[6:])
        except json.JSONDecodeError:
            return None
    return None
```

This is the same parsing any SSE client would do (browser `EventSource`, curl, etc.). The `[6:]` slice removes the 6-character `data: ` prefix.

### The submit handler -- three event types, three behaviors

```python
async def on_submit():
    ctx = AppContext(db=db_instance, user_email=email)

    async for raw_event in run_triage_stream_events(ctx, query):
        log.push(raw_event.strip())       # Always push to raw log
        payload = parse_sse_event(raw_event)
        if payload is None:
            continue

        if "status" in payload:
            # Tool callback events -- update the status label progressively
            # These arrive during the tool call phase (classify, lookup, escalate)
            status_label.set_text(f"⚙ {payload['status']}")

        elif "customer_reply" in payload:
            # Partial token events -- silently accumulate, don't show yet
            # The reply card stays hidden; we wait for the complete final text
            pass

        elif "final" in payload:
            final = payload["final"]

            # Reveal reply card with completed text
            reply_label.set_text(final.get("customer_reply", ""))
            reply_card.classes(remove="hidden")

            # Populate and reveal result card
            category = final.get("category", "unknown")
            category_val.set_text(category.replace("_", " ").title())
            # ... badges, order id, suggested action ...
            result_card.classes(remove="hidden")

            # Final status
            status_label.set_text("✓ Done")
```

**Why ignore `customer_reply` partial events?** The `customer_reply` tokens stream in during the short window after all tool calls complete. Because the orchestrator front-loads all its work (tool calls) before streaming the final output, the partial outputs arrive in a burst and are almost immediately followed by `final`. Waiting for `final` gives a clean, complete reply rather than a flickering partial text. If you were streaming a long free-text response (not structured output), streaming partials to the user would make more sense.

### Keeping reply and result cards hidden until done

On each new submission, both cards are explicitly hidden before the agent runs:

```python
# Reset on every new submission
reply_card.classes(add="hidden")
result_card.classes(add="hidden")
status_label.set_text("⚙ Starting agent...")
```

They are only revealed inside the `elif "final"` branch. This means the user never sees a stale result from the previous run while the new one is loading.

### Status label color transitions

The status label starts as slate (neutral), and turns green when done:

```python
# While running: slate gray
status_label.classes("text-slate-400")
status_label.set_text("⚙ Starting agent...")

# On final event: green
status_label.classes(remove="text-slate-400")
status_label.classes("text-green-400")
status_label.set_text("✓ Done")
```

`classes(remove="...")` and `classes("...")` are NiceGUI's way to toggle Tailwind classes at runtime. The pattern is: remove the old color class first, then add the new one.

### Category color coding

Each `RequestCategory` value maps to a distinct badge color so the result is scannable at a glance:

```python
CATEGORY_COLORS = {
    "refund": "#f97316",           # orange -- high-stakes, needs attention
    "technical_support": "#3b82f6", # blue -- informational
    "general_query": "#22c55e",    # green -- low priority
    "unknown": "#ef4444",          # red -- agent failure
}
```

### Running the UI

```bash
python ui.py
# Opens at http://localhost:8080
```

The UI runs on port 8080 to avoid colliding with the FastAPI server on port 8000. Both can run simultaneously -- they're independent servers that happen to share the same Python service layer.

---

## Structured Logging

`print()` is a one-way street -- text goes to the terminal and disappears. Python's `logging` module gives you severity levels, timestamps, per-module filtering, and file output with zero extra dependencies.

| Feature | `print()` | `logging` |
|---|---|---|
| Severity levels | None -- all equal | DEBUG / INFO / WARNING / ERROR |
| Timestamps | No | Yes, automatic |
| Filter by severity | No | Yes -- set `INFO` in prod, `DEBUG` in dev |
| Per-module control | No | Yes -- silence noisy third-party libraries |
| File output | No | Yes, can write to `.log` files |

### Setup: one shared logger factory

Create `src/logger.py` as the single place where logging is configured. Every other file imports from here:

```python
import logging
import sys

def get_logger(name: str) -> logging.Logger:
    """Get a logger with a specific name."""
    logger = logging.getLogger(name)

    # Prevent adding duplicate handlers if get_logger is called multiple times
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

    return logger
```

The `if not logger.handlers:` guard is important. Python's `logging` module caches logger instances by name -- calling `logging.getLogger("src.triage_service")` twice returns the same object. Without the guard, each call to `get_logger` would add another handler to the same logger, and every log line would print twice (or more).

### Using the logger in each file

```python
from src.logger import get_logger

logger = get_logger(__name__)
```

`__name__` is a Python built-in that equals the current module's fully-qualified name (e.g. `src.triage_service`, `api`). Every log line will identify exactly which file it came from:

```
2026-02-18 10:06:21 - src.triage_service - INFO - Orchestrator run complete | ...
2026-02-18 10:06:22 - api - INFO - Triage request received | email=user1@gmail.com
```

### Severity levels -- use the right one

```python
logger.debug("Detailed internal state, only useful when actively debugging")
logger.info("Normal operation -- request received, run complete, cost logged")
logger.warning("Something unexpected but recoverable -- fallback used, retry triggered")
logger.error("Something broke -- orchestrator failed, exception caught")
```

Use `INFO` for normal operational events you always want to see. Use `ERROR` only for actual failures. Never use `ERROR` for expected paths (like a customer providing a non-existent order number -- that's a `WARNING` at most, not an error).

### Structured log messages -- embed the data in the string

The `logging` module supports an `extra={}` dict that attaches fields to the log record. However, with a standard formatter like `%(asctime)s - %(name)s - %(levelname)s - %(message)s`, those fields are **silently dropped** because the format string doesn't reference them by name.

The reliable pattern is to embed the data directly in the message string using f-strings:

```python
# BAD: extra={} fields are invisible with standard formatters
logger.info("Orchestrator run complete", extra={"category": output.category.value})

# GOOD: all data visible in the log line
usage = result.usage()
logger.info(
    f"Orchestrator run complete | user={ctx.user_email} | category={output.category.value} | "
    f"input_tokens={usage.input_tokens} | output_tokens={usage.output_tokens} | requests={usage.requests}"
)
```

Output:
```
2026-02-18 10:06:21 - src.triage_service - INFO - Orchestrator run complete | user=user1@gmail.com | category=refund | input_tokens=2189 | output_tokens=222 | requests=4
```

The `extra={}` approach is useful in production systems that ship logs to aggregators (Datadog, Grafana Loki) which parse JSON fields. For terminal output, the f-string pattern is clearer and safer.

### What to log and what not to log

| Location | Use | Why |
|---|---|---|
| `triage_service.py` | `logger.info` for run complete (with usage), `logger.error` for failures | Core operational events and cost tracking |
| `api.py` | `logger.info` for request received/complete, `logger.warning` for 503 triggers | HTTP-level visibility |
| `agents.py` | `logger.debug` inside tools if debugging a specific issue | Too noisy for permanent INFO logs |
| `main.py` | Keep `print()` | These are intentional CLI output for the developer, not log events |

The `print()` calls in `main.py` ("Dear client:", "Admin, do you approve?") are **not log messages** -- they are the user interface of the CLI. Replacing them with `logging` would be wrong; they'd get swallowed when the log level is set to WARNING.

### Log usage after every orchestrator run

Token cost is the most operationally important metric in an AI system. Log it after every run:

```python
usage = result.usage()
logger.info(
    f"Orchestrator run complete | user={ctx.user_email} | category={output.category.value} | "
    f"input_tokens={usage.input_tokens} | output_tokens={usage.output_tokens} | requests={usage.requests}"
)
```

`result.usage()` returns the **aggregate** across all delegate agents when you pass `usage=ctx.usage` to every delegate call. This single log line shows you the full cost of one customer request -- orchestrator + classifier + specialist + escalation combined.

---

## Interview Prep

This section prepares you to explain this project clearly and confidently in a technical interview. Each question includes a short answer you can say out loud and a deeper explanation for follow-up questions.

---

### "Walk me through this project."

**Short answer (30 seconds):**
> "I built a customer support triage system using Pydantic AI. Customers send in queries via HTTP. An orchestrator LLM decides which specialist agents to call -- classifier, support specialist, or escalation manager. Python code then enforces business rules on the result. The system supports both standard JSON responses and real-time SSE streaming. It's wrapped in a FastAPI API, a NiceGUI browser UI, and a CLI for local testing."

**What makes it interesting to talk about:**
- Multi-agent delegation, not hardcoded routing
- LLM decides, Python enforces (business rules live in code, not prompts)
- Three entry points (API, UI, CLI) sharing the same service layer
- Real-world challenges: LLM hallucinations, streaming latency, cost control

---

### "Why Pydantic AI instead of LangChain or LlamaIndex?"

**Short answer:**
> "Pydantic AI is Pythonic and minimal. It gives you type-safe structured output, native Pydantic model validation, and a clean dependency injection system -- without the abstraction layers that LangChain adds. I can read the Pydantic AI source and understand exactly what happens. With LangChain, there are so many layers of abstraction that debugging a bug can take you through five library files before you find the root cause."

**Deeper:**
- LangChain is powerful but heavily abstracted. It has its own concepts (chains, runnables, callbacks, LCEL) that you have to learn on top of Python and the LLM API.
- Pydantic AI is built on top of Pydantic, which most Python engineers already know. The structured output is literally a Pydantic `BaseModel` -- no extra "schema definition language" to learn.
- The dependency injection (`RunContext[AppContext]`) is type-safe. Your IDE gives you autocomplete inside tools and validators. LangChain's callback/chain system is much harder to trace statically.
- For production systems where you need to understand every step, less abstraction is better.

---

### "Why agent delegation instead of hardcoded routing?"

**Short answer:**
> "Hardcoded routing uses Python if/else to decide which agent to call based on keywords or categories. Agent delegation lets the orchestrator LLM decide, based on what it read. This handles edge cases better -- a query like 'I have a technical problem AND I want a refund' would fail to route cleanly with hardcoded logic. The orchestrator can decide to call both the specialist and the escalation agent."

**Deeper:**
- We actually started with programmatic hand-off (Python explicitly calls classifier, then specialist, then escalation based on the classifier's output). It works, but it's rigid. The routing logic is in your Python code, not in the LLM that understands language.
- With delegation, the orchestrator agent receives all available tools (classify, handle_support, escalate), reads the user's query, and decides the sequence itself. New tools can be added without changing the routing logic.
- The tradeoff: delegation uses more tokens and is harder to trace, since the LLM's reasoning is not visible. For predictable, simple flows, programmatic hand-off is more reliable. For complex, unpredictable inputs, delegation is more flexible.

---

### "How do you handle LLM hallucinations?"

**Short answer:**
> "I split the problem into two categories. For quality issues -- vague replies, too-short answers -- I use `ModelRetry` in output validators to re-prompt the model and ask it to do better. For factual issues that only my code can verify -- like whether an order ID exists in the database -- I override the LLM's output directly in Python after the run completes. The LLM cannot know what's in my database. Python can."

**Deeper:**
- This distinction took a real bug to learn. I had a case where a user mentioned a non-existent order ID. The LLM would extract the order ID (correctly), then confidently describe what it would do for that order (hallucinating that the order existed).
- I first tried using `ModelRetry` inside the output validator to re-prompt the model to set `order_id = null`. But the model kept extracting the same ID from the user's message -- because the user said it, and from the model's perspective it was doing the right thing. The retry loop exhausted and triggered the generic fallback.
- The real fix: don't ask the LLM to re-decide. In `apply_business_rules()`, after the orchestrator run completes, check if the extracted `order_id` exists in the database. If it doesn't, construct a `FinalTriageResponse` directly in Python with the right `customer_reply` and return early. The LLM never sees this -- Python overrides it entirely.
- Rule: **`ModelRetry` for quality issues the LLM can fix. Deterministic override for facts the LLM cannot know.**

---

### "How do you handle failures? What happens when the LLM API is down?"

**Short answer:**
> "There are three layers of error handling. First, each delegate tool has a try/except that returns a descriptive error string instead of raising. The orchestrator sees the error message and can adapt. Second, the triage service has a try/except around the orchestrator run that returns a fallback `FinalTriageResponse` with `category='unknown'`. Third, the API layer detects `category='unknown'` and converts it to an HTTP 503, so the client knows the service is degraded."

**Deeper:**
- Layer 1 (tool level): `try/except` in `classify_request`, `handle_support_request`, `escalate_to_manager`. Each returns a string like `"ERROR: Escalation failed: {error}"`. The orchestrator reads this and can flag for manual review.
- Layer 2 (service level): `run_triage` and `run_triage_stream_events` both have outer `try/except` that catch any uncaught exception (connection errors, token limit exceeded, model timeout) and return a safe fallback response.
- Layer 3 (API level): `api.py` checks if `result.category == RequestCategory.UNKNOWN`. If so, it raises `HTTPException(503)` with detail information so the caller knows it's a server-side issue, not bad input.
- The `UNKNOWN` enum value is the sentinel that propagates through all three layers without requiring special exception types.

---

### "How do you manage costs? What prevents runaway token usage?"

**Short answer:**
> "`UsageLimits` on every agent run. I set both a `request_limit` (max LLM API calls per run, to prevent retry loops) and a `total_tokens_limit` (hard cap on tokens). I also pass `usage=ctx.usage` from delegate agents to the orchestrator, so I get a single aggregate usage number for the entire chain -- not just what the orchestrator used."

**Deeper:**
- Without `UsageLimits`, a `ModelRetry` loop can call the API dozens of times before crashing. With `request_limit=10`, it hard-stops at 10 calls.
- The `usage=ctx.usage` pattern is how cost roll-up works. When you call a delegate agent inside a tool, passing the parent's usage object tells Pydantic AI to add all tokens used by the delegate to the parent's counter. One call to `result.usage()` at the end gives you the total cost across all agents.
- In logging, after every orchestrator run, we log the usage: `logger.info(f"user={ctx.deps.user_email} category={output.category.value} usage={result.usage()}")`. This means every request is traceable by user and cost.
- Model selection also matters: the classifier uses a local Ollama model (free, zero latency, fast) for the simple categorization task. Only complex reasoning uses the OpenAI cloud model.

---

### "How do you test an AI system? You can't unit test a language model."

**Short answer:**
> "You don't test the LLM -- you test everything around it. Business rules are pure Python functions that don't call any LLM. I can run 18 unit tests in milliseconds that verify every combination of category, order existence, and edge case in `apply_business_rules`. Output validators are also pure Python -- I use `MagicMock` to fake a `RunContext` and test that the validator correctly raises `ModelRetry` for bad inputs and passes good inputs."

**Deeper:**
- The core philosophy: separate what the LLM decides from what Python enforces. The LLM parts are expensive and non-deterministic -- you can't reliably unit test them. The Python parts (business rules, validators) are deterministic and free to test.
- `tests/test_business_rules.py` has 9 tests covering: refund forces approval, general query needs no approval, non-existent order triggers override, tech support with known order preserves LLM judgment, and data integrity (other fields not affected by business rules).
- `tests/test_validators.py` has 9 tests covering: valid output passes all validators, each rejection condition (short reply, vague action, invalid severity, short memo), and the streaming guard (`ctx.partial_output = True` skips validation).
- `MagicMock` for `RunContext` is the key technique. You build a mock context with the data your validator needs, then call the validator directly. No agent, no LLM, no API key needed.

---

### "How does streaming work? Both at the agent level and at the API level?"

**Short answer:**
> "At the agent level, `agent.run_stream()` returns tokens as they arrive from the LLM. You call `result.stream_output()` to get partial structured output objects as they build up. At the API level, we use FastAPI `StreamingResponse` with Server-Sent Events -- the client gets newline-delimited JSON objects. We also add status events during tool calls, so the client knows what the agent is doing while it waits."

**Deeper:**
- The orchestrator pattern creates a latency problem: the real work (tool calls to classifier, specialist, escalation) happens before the orchestrator starts generating its structured output. So 90% of the time, the client is waiting with no data. That's why status events matter.
- Status events use a callback pattern. `AppContext.on_status` is an optional async function. Delegate tools call it when they start: `await ctx.deps.on_status("Classifying request...")`. This function puts the message into an `asyncio.Queue`. The SSE generator drains the queue before yielding each partial output event.
- Three event types: `status` (tool call progress), `customer_reply` (partial structured output during final streaming window), `final` (complete structured output).
- The NiceGUI UI only shows the `customer_reply` card when the `final` event arrives. Partial reply events are silently ignored -- this prevents flickering or incomplete text appearing in the UI.

---

### "What's the hardest bug you fixed in this project?"

**Short answer:**
> "A business rule interaction bug combined with an LLM retry loop. A user asked for a refund on a non-existent order ID. My rule for 'non-existent orders' set `requires_human_approval = False`. My rule for 'refunds' set it to `True`. Both rules ran sequentially, and the second one always won -- a 'last-writer-wins' bug. I tried to fix it with `ModelRetry` in the validator to tell the LLM to stop extracting the order ID. But the LLM kept extracting it because the user literally said it. The retry loop exhausted and triggered the fallback error response."

**What I learned:**
- Business rules that modify the same field must be aware of each other's conditions, or one will silently overwrite the other.
- `ModelRetry` cannot fix contradictions between what the user said and what your business rules want. If the user said `#123`, the LLM will always extract `#123`. You can't retry your way to a different extracted value.
- The real fix: when a specific business rule fires (order not found), skip all remaining rules and return a complete `FinalTriageResponse` immediately. Early return, not flag mutation.

---

### "Would this work at scale? What would you change for production?"

**Short answer:**
> "The core architecture would hold up. The things I'd change are infrastructure, not design: replace the mock database with a real one (SQLite or PostgreSQL), add authentication to the API, route logs to a structured log aggregator (Datadog, CloudWatch), add integration tests that call the real agent with a real LLM, and containerize with Docker for reproducible deploys."

**What's already production-ready:**
- The service layer (`triage_service.py`) is completely decoupled from the HTTP layer. You can swap the API without changing any business logic.
- The agent outputs are fully typed Pydantic models. There's no JSON parsing in the service layer -- it's all type-safe from agent output to HTTP response.
- Layered error handling means no uncaught exceptions reach the client.
- The logging setup captures user, category, and token usage for every request.
- The `UNKNOWN` sentinel + HTTP 503 pattern means monitoring systems can detect agent failures without parsing error messages.

**What would need to change:**
- `MockDB` → real database with connection pooling
- No authentication currently (API is open)
- Logging goes to stdout -- fine for Docker, but you'd want a log shipper for aggregation
- No integration tests (testing with real LLM calls, not just mocks)
- No rate limiting on the API endpoints

---

### "Why is the classifier using Ollama (local model) while the orchestrator uses OpenAI?"

**Short answer:**
> "Classification is a simple, low-stakes task: put a message into one of four categories. A small local model handles this reliably and for free. The orchestrator needs to reason about which agents to call, combine their outputs, and produce a thoughtful customer reply -- that requires a capable cloud model. Using the right model for the right task keeps costs low without sacrificing quality where it matters."

**Deeper:**
- Local Ollama models (`llama3.2`) need `PromptedOutput(CustomerRequestResult)` instead of the plain `CustomerRequestResult` type, because smaller local models don't reliably follow the function-calling API for structured JSON. `PromptedOutput` wraps the schema as instructions in the system prompt instead, which smaller models handle better.
- Cloud models (OpenAI `gpt-4.1-mini`) understand the function-calling protocol natively, so you pass the schema directly as `output_type=FinalTriageResponse`.
- The cost calculation: classification runs on every single request. If classification uses a cloud model at $0.001/call and you have 10,000 requests/day, that's $10/day just for classification. A local model makes that $0.

---

### "Explain the AppContext design. Why is it a dataclass instead of a Pydantic model?"

**Short answer:**
> "`AppContext` is internal infrastructure -- it's never serialized, never sent over HTTP, never received from a user. Pydantic `BaseModel` adds validation overhead and is designed for data that arrives from untrusted sources (users, APIs, files). `dataclass` is lighter: it's just a container. The database instance, the user email string, and the callback function are all set directly by trusted Python code. There's nothing to validate."

**When to use `dataclass` vs `BaseModel`:**
- `BaseModel`: for data arriving from outside your code (HTTP request body, JSON files, LLM output). Validates, coerces, and rejects bad input.
- `dataclass`: for internal data structures that your own code creates and controls. No validation needed, no serialization overhead.

---

## Blueprint: Create Any Agent from a Description

Follow this blueprint to create a new agent for any task. This is the process a master agent should follow when building agents from a description.

### Input: Task description

Example: "Create an agent that reads a document and summarizes it."

### Process:

**1. Define the output schema**
Ask: What does the calling code need from this agent? Define those fields.

```python
class DocumentSummary(BaseModel):
    """Summary of a document"""
    title: str = Field(description="The title or subject of the document")
    summary: str = Field(description="A concise 2-3 sentence summary of the document's key points")
    key_topics: list[str] = Field(description="List of main topics covered in the document")
    sentiment: str = Field(description="Overall sentiment: 'positive', 'negative', or 'neutral'")
```

**2. Define dependencies**
Ask: What external data or services does the agent need?

```python
@dataclass
class SummaryContext:
    document_text: str       # The document content to summarize
    max_summary_length: int  # Configurable summary length
```

**3. Create the agent**

```python
summary_agent = Agent(
    model="openai:gpt-4.1-mini",
    output_type=DocumentSummary,
    deps_type=SummaryContext,
    retries=2
)
```

**4. Add dynamic system prompt**

```python
@summary_agent.system_prompt
def build_prompt(ctx: RunContext[SummaryContext]) -> str:
    return f"""
        You are a document analysis specialist.
        Summarize the following document in no more than {ctx.deps.max_summary_length} words.
        Focus on key findings, conclusions, and actionable items.
    """
```

**5. Add tools if the agent needs to perform actions**

Only add tools if the agent needs to DO something beyond analyzing the prompt.
A summarization agent probably doesn't need tools -- the document is in the context.
A research agent would need tools to search databases or fetch URLs.

**6. Add output validation**

```python
@summary_agent.output_validator
def validate_summary(ctx: RunContext[SummaryContext], output: DocumentSummary) -> DocumentSummary:
    if len(output.summary.split()) > ctx.deps.max_summary_length:
        raise ModelRetry(
            f"Summary is too long ({len(output.summary.split())} words). "
            f"Keep it under {ctx.deps.max_summary_length} words."
        )
    if len(output.key_topics) == 0:
        raise ModelRetry("You must identify at least one key topic from the document.")
    return output
```

**7. Run with usage limits (standard)**

```python
result = await summary_agent.run(
    user_prompt=document_text,
    deps=SummaryContext(document_text=document_text, max_summary_length=100),
    usage_limits=UsageLimits(request_limit=5, total_tokens_limit=8000)
)
print(result.output.summary)
print(result.usage())
```

**8. Run with streaming (if user-facing)**

```python
async with summary_agent.run_stream(
    user_prompt=document_text,
    deps=SummaryContext(document_text=document_text, max_summary_length=100),
    usage_limits=UsageLimits(request_limit=5, total_tokens_limit=8000)
) as result:
    async for partial in result.stream_output():
        if partial.summary:
            print(f"\r{partial.summary}", end="", flush=True)
    print()
    output = await result.get_output()
    print(result.usage())
```

### Checklist for any new agent

- [ ] Output schema defined with Field descriptions
- [ ] Dependency context defined with shared resources and per-request data
- [ ] Agent created with model, output_type, deps_type, retries
- [ ] Dynamic system prompt if runtime data is needed
- [ ] Tools added with @agent.tool if the agent needs to perform actions
- [ ] Output validator with ModelRetry for quality enforcement
- [ ] Output validator includes `if ctx.partial_output: return output` guard for streaming compatibility
- [ ] Business rule guardrails in application code (not in the LLM)
- [ ] Business rules extracted into a shared function if multiple run modes exist (standard + streaming)
- [ ] UsageLimits set on every agent.run() and agent.run_stream() call
- [ ] result.usage() tracked for cost monitoring (after get_output() for streaming)
- [ ] Non-reasoning model chosen unless the task genuinely requires deep reasoning
- [ ] Streaming support added if the agent is user-facing (run_stream + stream_output)
- [ ] `logger = get_logger(__name__)` added to every service/API file (not main.py CLI)
- [ ] Usage logged after every orchestrator run with user, category, and token counts
- [ ] `logger.error` used in every `except` block that catches orchestrator/agent failures
- [ ] Unit tests for business rules -- every combination of category, order existence, and edge cases
- [ ] Unit tests for output validators -- happy path, each rejection rule, and streaming guard (`ctx.partial_output = True`)
- [ ] Data integrity test -- verify non-approval fields are preserved through business rules

### Additional checklist for delegate agents (used inside orchestrator tools)

- [ ] Delegate tool function returns a string (use `model_dump_json()` for structured output)
- [ ] `usage=ctx.usage` passed to roll up token costs to the parent orchestrator
- [ ] `deps=ctx.deps` passed if the delegate needs dependency access (DB, user email)
- [ ] `try/except` inside delegate tool functions to prevent crashing the orchestrator
- [ ] Error returns are descriptive strings the orchestrator can reason about
- [ ] Orchestrator system prompt clearly describes available tools and the expected workflow
- [ ] All manual `FinalTriageResponse` constructors (especially in `except` blocks) include every required field
- [ ] Status callback (`if ctx.deps.on_status: await ctx.deps.on_status(...)`) added to each delegate tool for streaming UX

### Additional checklist for API streaming endpoints

- [ ] `on_status` callable added to `AppContext` as an optional field (defaults to `None`)
- [ ] `asyncio.Queue` created in the SSE generator to bridge between tool callbacks and yields
- [ ] `emit_status` function defined and stored in `AppContext.on_status`
- [ ] Queue drain loops placed at every yield point in the generator (before and inside the stream loop)
- [ ] Generator yields three event types: `status`, `customer_reply`, and `final`
- [ ] Error handler in the generator yields a `final` event with fallback response (not just an exception)
- [ ] `StreamingResponse` with `media_type="text/event-stream"` in the FastAPI endpoint
- [ ] Tested with `curl -N` (not Swagger UI, which doesn't handle SSE properly)

### Additional checklist for NiceGUI UI

- [ ] `parse_sse_event()` helper strips `data: ` prefix and parses JSON from each generator yield
- [ ] `status` events update the status label only (tool call progress feedback during the slow phase)
- [ ] `customer_reply` partial events are silently ignored -- UI waits for `final` to show complete text
- [ ] `final` event reveals both the reply card and result card simultaneously
- [ ] Both cards are explicitly hidden (`classes(add="hidden")`) at the start of every new submission
- [ ] Status label color transitions: slate (running) → green (done) via `classes(remove=...)` + `classes(...)`
- [ ] Category badges are color-coded: orange (refund), blue (technical), green (general), red (unknown)
- [ ] Raw event log at the bottom receives every event via `log.push()` for debugging
- [ ] UI runs on a different port (8080) than the FastAPI server (8000) to avoid conflicts

---

## File Structure Reference

```
automated-ai-customer-support/
├── api.py                   # FastAPI HTTP layer -- /triage, /triage/stream, /health endpoints
├── main.py                  # CLI entry point -- for local testing and human-in-the-loop
├── ui.py                    # NiceGUI interactive browser UI -- real-time agent output display
├── src/
│   ├── agents.py            # Agent definitions, system prompts, tools, validators
│   ├── config.py            # AppContext, model names, constants, feature flags
│   ├── logger.py            # Shared logger factory -- get_logger(__name__) for every module
│   ├── schemas.py           # Pydantic schemas (agent output + API request/response)
│   ├── triage_service.py    # Business logic, agent orchestration, guardrails, streaming
│   └── db.py                # Database layer (mock or real)
├── tests/
│   ├── test_business_rules.py  # Tests for apply_business_rules() -- deterministic rule verification
│   └── test_validators.py      # Tests for output validators -- ModelRetry and streaming guard verification
├── _learning/
│   └── PYDANTIC_AI_AGENT_GUIDE.md  # This document
├── pyproject.toml           # Dependencies
└── .env                     # API keys (never commit)
```

The separation:
- `schemas.py` defines WHAT agents return AND what the API accepts (`TriageRequest`, `FinalTriageResponse`, `EscalationResponse`). Pydantic models serve double duty -- agent output schema and API contract.
- `agents.py` defines HOW agents work -- all 4 agents live here (classifier, specialist, escalation, orchestrator) along with their system prompts, tools, and validators
- `triage_service.py` defines WHEN and WHY agents are called (calls the orchestrator, applies business rules, handles both standard and streaming run modes)
- `config.py` defines configuration (models, limits, feature flags like `IS_STREAM_RESPONSE_OUTPUT`)
- `logger.py` is the shared logging factory -- all modules call `get_logger(__name__)` to get a consistently formatted logger. The `if not logger.handlers` guard prevents duplicate output when the same module is imported multiple times
- `api.py` is the HTTP interface -- receives requests, creates per-request contexts, calls `run_triage` or `run_triage_stream_events`, and translates agent failures into proper HTTP status codes. Includes both standard JSON endpoint and SSE streaming endpoint
- `main.py` is the CLI interface -- for local testing, `asyncio.gather` parallel runs, and human-in-the-loop approval
- `ui.py` is the NiceGUI browser UI -- consumes `run_triage_stream_events` directly, displays status events as live progress and the final response once complete
- `tests/` verifies deterministic components (business rules, validators) without LLM calls -- fast, free, reliable

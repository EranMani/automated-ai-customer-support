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
16. [Blueprint: Create Any Agent from a Description](#blueprint-create-any-agent-from-a-description)

---

## Core Philosophy

These principles govern every design decision in this codebase:

1. **LLMs decide, Python enforces.** Use the LLM for what it is good at: understanding natural language, interpreting intent, generating human-like responses. Enforce business rules deterministically in Python code, never in the LLM.

2. **Structured output is the contract.** The output schema is the bridge between the LLM's reasoning and your application logic. Every piece of data your code needs to act on must be a field in the output schema.

3. **Don't trust, verify.** LLMs are probabilistic. The same input can produce different outputs. Always validate LLM output against your own data sources (database, APIs) before acting on it.

4. **Dependencies flow through context.** Per-request data (user identity, session) lives in the dependency context. Shared resources (database, HTTP clients) are passed through the same context but instantiated once.

5. **Co-locate agent capabilities.** An agent's system prompt, tools, and output validator should live together in one place. Anyone reading the code should understand the agent's full capability set at a glance.

6. **Let agents delegate, let Python override.** An orchestrator agent decides which specialist agents to call. But business rules (refund approval, order validation) are always enforced by Python code on the final output, never left to the LLM's judgment.

---

## Architecture Overview

This project uses an **agent delegation** pattern. An orchestrator agent is the brain that decides which specialist agents to call. Python code enforces business rules on the final output.

```
User Request
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
```

Key design decisions:
- **Agent delegation**: The orchestrator LLM decides which agents to call and in what order, instead of hardcoded Python `if/else` routing
- **Multiple models**: Cheap local model (Ollama) for classification, capable cloud model (OpenAI) for complex reasoning and orchestration
- **Structured output at every stage**: Classifier returns `CustomerRequestResult`, specialist returns `FinalTriageResponse`, escalation returns `EscalationResponse`
- **Business rules in Python**: Refund approval, order existence checks, and approval flags are enforced by code, never by the LLM's judgment
- **Usage roll-up**: All delegate agents pass `usage=ctx.usage` so the orchestrator tracks total cost across the entire chain

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

@dataclass
class AppContext:
    db: MockDB          # Shared resource -- one instance for all requests
    user_email: str     # Per-request data -- different for each user
```

Key distinction:
- **Shared resources** (database connections, HTTP clients): Created once, passed to every context
- **Per-request data** (user email, session ID): Different for each agent run, set when creating the context

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

### Design principles

1. **Every field the application needs must be in the schema.** If your refund logic needs an order ID, the schema must include `order_id`. The LLM identifies it from the conversation; your code uses it to process the refund.

2. **Field descriptions are instructions to the LLM.** The `description` parameter in `Field()` is sent directly to the model. Write it like you are instructing a person: be specific about format, constraints, and expectations.

3. **Use `str | None` for conditional fields.** Not every request involves an order. Making `order_id` optional with `default=None` lets the LLM skip it for general queries.

4. **Don't include fields you can verify yourself.** If you can check whether an order exists by querying your database, don't add `is_order_found: bool` to the schema. The LLM might get it wrong. Verify it in your Python code instead.

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

If a value is known from context, don't make the LLM provide it as a parameter:

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

The fewer parameters the LLM must fill in, the fewer chances for errors.

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

How it works:
1. LLM returns `FinalTriageResponse`
2. Pydantic validates the JSON against the schema (catches type errors, missing fields)
3. Your output validator runs (catches quality issues)
4. If `ModelRetry` is raised, the error message is sent back to the LLM as feedback
5. The LLM tries again (up to the `retries` limit on the agent)
6. If all retries fail, an exception is raised

Use `ModelRetry` for problems the model CAN fix: vague responses, missing detail, wrong format.
Do NOT use it for business logic -- use deterministic overrides instead (see next section).

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

# Business rule: verify the order actually exists in our system
if output.order_id is not None:
    try:
        ctx.db.get_order_status(output.order_id)
    except KeyError:
        # Order doesn't exist in DB -- nothing to approve
        requires_human_approval = False

# Business rule: no order = nothing to approve
if output.order_id is None:
    requires_human_approval = False
```

### The principle

Don't ask the LLM to tell you something you can verify yourself:
- The LLM says "order #999 needs a refund" --> You check the database: #999 doesn't exist --> Override to False
- The LLM says "no approval needed" for a refund --> Your business rule says refunds always need approval --> Override to True

**LLM provides**: intent interpretation, natural language responses, entity extraction
**Python enforces**: business rules, data validation, access control

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

    # Business rule: verify the order actually exists in the system
    if output.order_id is not None:
        try:
            ctx.db.get_order_status(output.order_id)
        except KeyError:
            requires_human_approval = False

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

### Understanding token costs in agentic systems

Every tool call creates a round-trip. Every round-trip resends the full conversation history. Token usage compounds:
- Request 1: system prompt + user query (~300 tokens)
- Request 2: all of the above + tool call + tool result (~600 tokens)
- Request 3: all of the above + second tool call + result (~900 tokens)
- Request 4: all of the above + structured output attempt (~1200 tokens)

This is why the specialist uses ~2000 input tokens across 4 requests. It is not a bug -- it is how LLMs work (they are stateless; every request sends the full context).

### Model selection impacts cost

- **Reasoning models** (gpt-5-nano, o-series): Generate internal chain-of-thought tokens. A simple reply can use 1700+ reasoning tokens. Expensive for straightforward tasks.
- **Non-reasoning models** (gpt-4.1-mini, gpt-4o-mini): No reasoning token overhead. Use these for tasks that don't require deep multi-step thinking.

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

The delegate agent returns structured output internally (e.g., `FinalTriageResponse`). But the tool function must serialize it to a string for the orchestrator using `model_dump_json()`. The orchestrator reads the JSON string, interprets it, and incorporates it into its own response.

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
        return f"ERROR: Specialist agent failed: {str(e)}. Try a different approach."
```

**Rule 5: The orchestrator's system prompt guides but doesn't guarantee tool order.**

You can instruct the orchestrator to "ALWAYS classify first, then handle support." But the LLM might ignore this. It might skip classification and go straight to the specialist. It might call escalation without calling the specialist first. You cannot guarantee tool call order the same way you can with Python `if/else`. This is the fundamental trade-off of letting the LLM be the brain.

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

### Additional checklist for delegate agents (used inside orchestrator tools)

- [ ] Delegate tool function returns a string (use `model_dump_json()` for structured output)
- [ ] `usage=ctx.usage` passed to roll up token costs to the parent orchestrator
- [ ] `deps=ctx.deps` passed if the delegate needs dependency access (DB, user email)
- [ ] `try/except` inside delegate tool functions to prevent crashing the orchestrator
- [ ] Error returns are descriptive strings the orchestrator can reason about
- [ ] Orchestrator system prompt clearly describes available tools and the expected workflow
- [ ] All manual `FinalTriageResponse` constructors (especially in `except` blocks) include every required field

---

## File Structure Reference

```
automated-ai-customer-support/
├── main.py                  # Entry point, orchestration, human-in-the-loop
├── src/
│   ├── agents.py            # Agent definitions, system prompts, tools, validators
│   ├── config.py            # AppContext, model names, constants, feature flags
│   ├── schemas.py           # Pydantic output schemas (the contract)
│   ├── triage_service.py    # Business logic, agent orchestration, guardrails, streaming
│   └── db.py                # Database layer (mock or real)
├── _learning/
│   └── PYDANTIC_AI_AGENT_GUIDE.md  # This document
├── pyproject.toml           # Dependencies
└── .env                     # API keys (never commit)
```

The separation:
- `schemas.py` defines WHAT agents return (`CustomerRequestResult`, `FinalTriageResponse`, `EscalationResponse`)
- `agents.py` defines HOW agents work -- all 4 agents live here (classifier, specialist, escalation, orchestrator) along with their system prompts, tools, and validators
- `triage_service.py` defines WHEN and WHY agents are called (calls the orchestrator, applies business rules, handles both standard and streaming run modes)
- `config.py` defines configuration (models, limits, feature flags like `IS_STREAM_RESPONSE_OUTPUT`)
- `main.py` handles user interaction, creates per-user contexts, and manages the human-in-the-loop approval flow

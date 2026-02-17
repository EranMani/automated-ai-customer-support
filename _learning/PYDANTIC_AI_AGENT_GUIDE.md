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
17. [Blueprint: Create Any Agent from a Description](#blueprint-create-any-agent-from-a-description)

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
├── tests/
│   ├── test_business_rules.py  # Tests for apply_business_rules() -- deterministic rule verification
│   └── test_validators.py      # Tests for output validators -- ModelRetry and streaming guard verification
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
- `tests/` verifies deterministic components (business rules, validators) without LLM calls -- fast, free, reliable

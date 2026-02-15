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
10. [Usage Limits and Cost Control](#usage-limits-and-cost-control)
11. [Multi-Agent Orchestration](#multi-agent-orchestration)
12. [Model Selection Guidelines](#model-selection-guidelines)
13. [Common Mistakes to Avoid](#common-mistakes-to-avoid)
14. [Blueprint: Create Any Agent from a Description](#blueprint-create-any-agent-from-a-description)

---

## Core Philosophy

These principles govern every design decision in this codebase:

1. **LLMs decide, Python enforces.** Use the LLM for what it is good at: understanding natural language, interpreting intent, generating human-like responses. Enforce business rules deterministically in Python code, never in the LLM.

2. **Structured output is the contract.** The output schema is the bridge between the LLM's reasoning and your application logic. Every piece of data your code needs to act on must be a field in the output schema.

3. **Don't trust, verify.** LLMs are probabilistic. The same input can produce different outputs. Always validate LLM output against your own data sources (database, APIs) before acting on it.

4. **Dependencies flow through context.** Per-request data (user identity, session) lives in the dependency context. Shared resources (database, HTTP clients) are passed through the same context but instantiated once.

5. **Co-locate agent capabilities.** An agent's system prompt, tools, and output validator should live together in one place. Anyone reading the code should understand the agent's full capability set at a glance.

---

## Architecture Overview

```
User Request
    |
    v
[Classifier Agent] -- Local LLM (Ollama) -- Cheap, fast intent detection
    |
    |-- GENERAL_QUERY --> Automated FAQ response (no LLM needed)
    |
    |-- REFUND / TECHNICAL_SUPPORT
            |
            v
      [Specialist Agent] -- Cloud LLM (OpenAI) -- Complex reasoning
            |
            |-- Calls tools (fetch_user_tier, fetch_order_status)
            |-- Output validated by @output_validator
            |-- Business rules applied deterministically
            |
            v
      [Human Approval] -- If refund + order exists --> Admin confirms
            |
            v
      [Process Action] -- Database update
```

Key design decisions:
- **Two models**: Cheap local model for classification, capable cloud model for complex reasoning
- **Structured output at every stage**: Classifier returns `CustomerRequestResult`, specialist returns `FinalTriageResponse`
- **Business rules in Python**: Refund approval is enforced by code, not by the LLM's judgment

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

```python
@specialist_agent.output_validator
def validate_output(ctx: RunContext[AppContext], output: FinalTriageResponse) -> FinalTriageResponse:
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

### Step 7: Run the Agent

```python
from pydantic_ai import UsageLimits

specialist_response = await specialist_agent.run(
    user_prompt=user_query,
    deps=AppContext(db=db_instance, user_email="user@example.com"),
    usage_limits=UsageLimits(request_limit=10, total_tokens_limit=4000)
)

# Access the structured output
print(specialist_response.output.customer_reply)
print(specialist_response.output.order_id)

# Track usage for cost monitoring
print(specialist_response.usage())
```

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
```

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

The classifier tagged the request as a REFUND. The specialist agent might set `requires_human_approval=False` because the order status says "Refund Processing" and the LLM interprets that as "already handled." Your business rule says every refund needs human approval, regardless.

```python
requires_human_approval = specialist_response.output.requires_human_approval

# Business rule: refund intent always requires approval
if intent == RequestCategory.REFUND:
    requires_human_approval = True

# Business rule: verify the order actually exists in our system
if specialist_response.output.order_id is not None:
    try:
        ctx.db.get_order_status(specialist_response.output.order_id)
    except KeyError:
        # Order doesn't exist in DB -- nothing to approve
        requires_human_approval = False

# Business rule: no order = nothing to approve
if specialist_response.output.order_id is None:
    requires_human_approval = False
```

### The principle

Don't ask the LLM to tell you something you can verify yourself:
- The LLM says "order #999 needs a refund" --> You check the database: #999 doesn't exist --> Override to False
- The LLM says "no approval needed" for a refund --> Your business rule says refunds always need approval --> Override to True

**LLM provides**: intent interpretation, natural language responses, entity extraction
**Python enforces**: business rules, data validation, access control

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

## Multi-Agent Orchestration

This project uses two agents in a programmatic hand-off pattern:

```python
async def run_triage(ctx: AppContext, user_query: str) -> FinalTriageResponse:
    # Agent 1: Classify intent (cheap, local model)
    classifier_response = await classifier_agent.run(user_prompt=user_query)
    intent = classifier_response.output.category

    # Route based on classification
    if intent == RequestCategory.GENERAL_QUERY:
        return FinalTriageResponse(...)  # No LLM needed

    # Agent 2: Handle complex requests (capable cloud model)
    specialist_response = await specialist_agent.run(user_prompt=user_query, deps=ctx)

    # Apply business rules to the specialist's output
    ...
    return FinalTriageResponse(...)
```

### Why two agents?

- **Cost**: The classifier uses a free local model (Ollama). Only complex requests hit the paid OpenAI API.
- **Speed**: Local classification is fast. The specialist only runs when needed.
- **Separation of concerns**: The classifier interprets intent. The specialist handles the case. Each has a focused system prompt and output schema.

### Running agents in parallel

Use `asyncio.gather` to run independent agent calls concurrently:

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

**7. Run with usage limits**

```python
result = await summary_agent.run(
    user_prompt=document_text,
    deps=SummaryContext(document_text=document_text, max_summary_length=100),
    usage_limits=UsageLimits(request_limit=5, total_tokens_limit=8000)
)
print(result.output.summary)
print(result.usage())
```

### Checklist for any new agent

- [ ] Output schema defined with Field descriptions
- [ ] Dependency context defined with shared resources and per-request data
- [ ] Agent created with model, output_type, deps_type, retries
- [ ] Dynamic system prompt if runtime data is needed
- [ ] Tools added with @agent.tool if the agent needs to perform actions
- [ ] Output validator with ModelRetry for quality enforcement
- [ ] Business rule guardrails in application code (not in the LLM)
- [ ] UsageLimits set on every agent.run() call
- [ ] result.usage() tracked for cost monitoring

---

## File Structure Reference

```
automated-ai-customer-support/
├── main.py                  # Entry point, orchestration, human-in-the-loop
├── src/
│   ├── agents.py            # Agent definitions, system prompts, tools, validators
│   ├── config.py            # AppContext, model names, constants
│   ├── schemas.py           # Pydantic output schemas (the contract)
│   ├── triage_service.py    # Business logic, agent orchestration, guardrails
│   └── db.py                # Database layer (mock or real)
├── pyproject.toml           # Dependencies
└── PYDANTIC_AI_AGENT_GUIDE.md  # This document
```

The separation:
- `schemas.py` defines WHAT agents return
- `agents.py` defines HOW agents work (prompts, tools, validation)
- `triage_service.py` defines WHEN and WHY agents are called (business logic)
- `main.py` handles user interaction and top-level orchestration

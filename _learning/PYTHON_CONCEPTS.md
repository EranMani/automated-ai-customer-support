# Python Concepts Reference

This document covers every Python concept used in the `automated-ai-customer-support` project.
Each concept is explained from first principles with real examples from the codebase, and includes notes on what an interviewer is likely to ask about it.

---

## Table of Contents

1. [async / await and Coroutines](#1-async--await-and-coroutines)
2. [asyncio — Concurrency Without Threads](#2-asyncio--concurrency-without-threads)
3. [Async Context Managers (`async with`)](#3-async-context-managers-async-with)
4. [Generators and Async Generators (`yield`)](#4-generators-and-async-generators-yield)
5. [Decorators](#5-decorators)
6. [Type Hints](#6-type-hints)
7. [Callables as First-Class Values](#7-callables-as-first-class-values)
8. [Dataclasses](#8-dataclasses)
9. [Enums](#9-enums)
10. [Exception Handling](#10-exception-handling)
11. [f-strings](#11-f-strings)
12. [Optional Values and None](#12-optional-values-and-none)
13. [Python Logging Module](#13-python-logging-module)
14. [Testing with pytest](#14-testing-with-pytest)
15. [MagicMock — Faking Objects in Tests](#15-magicmock--faking-objects-in-tests)
16. [The Module System](#16-the-module-system)
17. [JSON Serialization](#17-json-serialization)

---

## 1. async / await and Coroutines

### What is it?

A **coroutine** is a function defined with `async def`. When you call it, it does NOT run immediately. It returns a coroutine object -- a suspended computation that you can resume later with `await`.

```python
# Normal function: runs immediately when called, blocks until it returns
def fetch_data():
    return requests.get("http://api.example.com")  # blocks for 500ms

# Coroutine: calling it returns a coroutine object, doesn't run yet
async def fetch_data():
    return await httpx.get("http://api.example.com")  # suspends while waiting

# To actually run the coroutine, you must await it:
result = await fetch_data()
```

### The mental model

`await` means: "Start this coroutine. While it's waiting for something external (network, disk, timer), **let the event loop run something else**. Come back here when the result is ready."

Without `await`: one request blocks the entire program while waiting for a response.
With `await`: while request A waits for a network response, request B can start, and request C can process.

### How it's used in this project

Almost every function in the service layer is `async`:

```python
# triage_service.py
async def run_triage(ctx: AppContext, user_query: str) -> FinalTriageResponse:
    result = await orchestrator_agent.run(   # suspends until the LLM responds
        user_prompt=user_query,
        deps=ctx,
    )
    return apply_business_rules(ctx, result.output)
```

```python
# agents.py -- async tool calling a delegate agent
@orchestrator_agent.tool
async def classify_request(ctx: RunContext[AppContext], customr_message: str) -> str:
    result = await classifier_agent.run(user_prompt=customr_message, usage=ctx.usage)
    return f"Category: {result.output.category.value}"
```

### Why it matters for this project

The FastAPI server uses an async event loop. Every request runs as a coroutine. When one request is waiting for the LLM API (which can take 2-5 seconds), the server can handle other incoming requests. Without `async/await`, every LLM call would block the server -- only one customer could be served at a time.

### Interview question: "What is the difference between `async def` and `def`?"

> `def` is a regular function that runs synchronously -- it blocks the caller until it returns. `async def` defines a coroutine. Calling it returns a coroutine object that does nothing until awaited. `await` suspends the current coroutine, hands control back to the event loop, and resumes when the result is ready. The key point: `await` doesn't block the thread -- it yields control cooperatively, which is how multiple coroutines can run concurrently in a single thread.

### Interview question: "Can you `await` inside a regular function?"

> No. `await` can only be used inside an `async def` function. If you try, Python raises a `SyntaxError`. Async is infectious -- once a function needs to await something, it must itself be `async`, and every caller must either `await` it or use `asyncio.run()` to start the event loop.

---

## 2. asyncio — Concurrency Without Threads

### What is it?

`asyncio` is Python's standard library for writing concurrent code using coroutines. It provides:
- An **event loop**: the scheduler that decides which coroutine runs next
- **`asyncio.run()`**: entry point -- starts the event loop and runs one coroutine
- **`asyncio.gather()`**: runs multiple coroutines concurrently and collects all results
- **`asyncio.Queue`**: a thread-safe queue for passing data between coroutines

### How it's used in this project

**`asyncio.run()` -- entry point in `main.py`:**
```python
if __name__ == "__main__":
    asyncio.run(main())   # starts the event loop and runs main()
```

**`asyncio.gather()` -- running four triage requests in parallel:**
```python
results = await asyncio.gather(
    triage_func(AppContext(db=db_instance, user_email=user1["email"]), user1["query"]),
    triage_func(AppContext(db=db_instance, user_email=user2["email"]), user2["query"]),
    triage_func(AppContext(db=db_instance, user_email=user3["email"]), user3["query"]),
    triage_func(AppContext(db=db_instance, user_email=user4["email"]), user4["query"]),
)
```

All four LLM requests are in-flight at the same time. The total wait time is the max of the four, not the sum.

**`asyncio.Queue` -- bridging tool callbacks and the SSE generator:**
```python
# triage_service.py
status_queue: asyncio.Queue[str] = asyncio.Queue()

async def emit_status(message: str):
    await status_queue.put(f"data: {json.dumps({'status': message})}\n\n")

# Tools call emit_status() --> puts a message in the queue
# The generator drains the queue --> yields the message to the client
while not status_queue.empty():
    yield await status_queue.get()
```

### `asyncio.Queue` vs a regular list

A regular list is not safe to share between coroutines that run in an event loop -- if a coroutine modifies the list while another reads it, you get inconsistent state. `asyncio.Queue` is designed for this: `put()` and `get()` are atomic operations that the event loop coordinates safely.

### Interview question: "What is `asyncio.gather` and when do you use it?"

> `asyncio.gather()` takes multiple coroutines and schedules them all to run concurrently in the same event loop. It returns a list of their results in the same order they were passed. Use it when you have independent async tasks that don't depend on each other's results. In this project, four users' queries are independent -- there's no reason to wait for user 1 to finish before starting user 2's request.

### Interview question: "asyncio vs threading -- what's the difference?"

> Threading runs code in multiple OS threads, which can truly run in parallel on multiple CPU cores. asyncio uses a single thread with cooperative multitasking -- coroutines take turns on one thread. asyncio is better for I/O-bound tasks (network calls, disk reads) because threads spend most of their time waiting, and the overhead of creating/switching threads adds up. For CPU-bound tasks (heavy computation), threading (or multiprocessing) is the right tool -- asyncio won't help because the CPU never yields.

---

## 3. Async Context Managers (`async with`)

### What is it?

A **context manager** is an object with `__enter__` and `__exit__` methods. `with` calls `__enter__` at the start and `__exit__` at the end -- even if an exception is raised. This guarantees cleanup.

An **async context manager** does the same thing, but its `__enter__` and `__exit__` are coroutines. You use `async with` instead of `with`.

```python
# Regular context manager -- file handle cleanup
with open("file.txt") as f:
    data = f.read()
# f.close() called automatically here, even if read() raises

# Async context manager -- releases resources after the stream is done
async with orchestrator_agent.run_stream(...) as result:
    async for partial in result.stream_output():
        print(partial.customer_reply)
# stream closed automatically here, even if an exception is raised
```

### How it's used in this project

**`agent.run_stream()` returns an async context manager:**
```python
# triage_service.py
async with orchestrator_agent.run_stream(
    user_prompt=user_query,
    deps=stream_ctx,
    usage_limits=UsageLimits(request_limit=10, total_tokens_limit=4000)
) as result:
    async for partial_output in result.stream_output():
        if partial_output.customer_reply:
            yield f"data: {json.dumps({'customer_reply': partial_output.customer_reply})}\n\n"

    output = await result.get_output()
```

The `async with` block ensures the streaming connection to the LLM API is properly closed when the block exits -- whether it completes normally, hits a token limit, or raises an exception.

### Why `result.get_output()` instead of `result.output`?

Inside `run_stream()`, the result object manages an open network connection. `result.output` would be the output if it's already available. But the stream might still be in progress. `await result.get_output()` explicitly waits for the stream to fully complete and returns the final validated output. Always use `get_output()` -- never access `.output` directly on a streaming result.

### Interview question: "What problem does a context manager solve?"

> Resource cleanup. Without a context manager, you need try/finally to guarantee cleanup: `try: f = open(...) / use f / finally: f.close()`. Context managers move that cleanup into a reusable protocol. The `with` statement guarantees `__exit__` runs, no matter what. For async resources (network connections, streaming API sessions), you need `async with` because the cleanup itself is async.

---

## 4. Generators and Async Generators (`yield`)

### What is a generator?

A **generator** is a function that uses `yield` instead of (or in addition to) `return`. Instead of computing all values at once and returning them as a list, a generator computes one value at a time, pauses, and hands the value to the caller. The caller decides when to ask for the next value.

```python
# Regular function: computes all values, returns them at once
def get_events():
    return ["event1", "event2", "event3"]   # all in memory at once

# Generator: yields one value at a time, pauses between each
def get_events():
    yield "event1"   # pauses here until caller asks for next
    yield "event2"   # pauses here
    yield "event3"
```

### Async generators

An **async generator** is a function with both `async def` and `yield`. It can `await` between yields. This is how you produce a sequence of values over time in an async context.

```python
async def run_triage_stream_events(ctx: AppContext, user_query: str) -> AsyncIterator[str]:
    # ...setup...
    async with orchestrator_agent.run_stream(...) as result:
        async for partial_output in result.stream_output():
            yield f"data: {json.dumps({'customer_reply': ...})}\n\n"
    
    yield f"data: {json.dumps({'final': output.model_dump()})}\n\n"
```

The `AsyncIterator[str]` return type annotation is how you declare "this function returns an async generator that produces strings one at a time."

### How the caller consumes it

```python
# api.py -- FastAPI
return StreamingResponse(
    run_triage_stream_events(ctx, request.query),  # passes the generator itself
    media_type="text/event-stream"
)

# ui.py -- NiceGUI
async for raw_event in run_triage_stream_events(ctx, query):
    log.push(raw_event.strip())
    payload = parse_sse_event(raw_event)
    # process each event as it arrives
```

`StreamingResponse` consumes the generator lazily -- it calls `__anext__()` to get one event, sends it to the client, then asks for the next one. The client receives events as they are generated, not all at once.

### `yield` vs `return` in context

```python
# return: exits the function and sends one value
def get_result():
    return "done"   # caller gets "done" and the function is gone

# yield: pauses the function and sends one value, function stays alive
def get_events():
    yield "step1"   # pauses, sends "step1", function is suspended
    yield "step2"   # pauses, sends "step2"
    yield "final"   # pauses, sends "final"
    # function ends here naturally
```

### Interview question: "Why use a generator instead of returning a list?"

> Two reasons: memory and latency. A list requires all values to be computed before any can be used. A generator computes one value at a time. For streaming responses, returning a list would mean buffering the entire response, then sending it all at once -- defeating the purpose of streaming. With a generator, each event is sent to the client the moment it's available. The client sees progress in real time instead of waiting for everything to finish.

### Interview question: "What is `AsyncIterator` and why is it the return type?"

> `AsyncIterator[str]` is the type annotation for "an async generator that produces strings." It tells the type checker and any caller: "you can use `async for event in this:` and you'll get strings." Using `AsyncIterator` as the return type is more honest than writing `-> None` because the function never `return`s a value -- it `yield`s them. `collections.abc.AsyncIterator` is the abstract base class that async generators satisfy automatically.

---

## 5. Decorators

### What is a decorator?

A decorator is a function that takes another function as input and returns a new (usually modified) function. The `@` syntax is shorthand for applying it.

```python
# These two are exactly equivalent:
@my_decorator
def my_function():
    pass

# Same as:
def my_function():
    pass
my_function = my_decorator(my_function)
```

Decorators are used to attach behavior to a function without modifying its body.

### Decorators used in this project

**`@dataclass` -- auto-generates `__init__`, `__repr__`, `__eq__` for a class:**
```python
@dataclass
class AppContext:
    db: MockDB
    user_email: str
    on_status: Callable[[str], Awaitable[None]] | None = None
```

Without `@dataclass`, you'd need to write `def __init__(self, db, user_email, on_status=None): self.db = db ...` manually.

**`@agent.tool` -- registers a function as an agent tool:**
```python
@specialist_agent.tool
def fetch_order_status(ctx: RunContext[AppContext], order_id: str) -> str:
    """Fetch the order status from the database."""
    ...
```

`@agent.tool` is a decorator method on the `Agent` object. When Python executes this `@` line, it calls `specialist_agent.tool(fetch_order_status)`, which registers the function internally so the agent knows it can call it.

**`@agent.system_prompt` -- registers the function as a dynamic prompt builder:**
```python
@specialist_agent.system_prompt
def read_database(ctx: RunContext[AppContext]) -> str:
    return f"You are a customer support specialist. The customer's email is: {ctx.deps.user_email}"
```

**`@agent.output_validator` -- registers the function as a post-run validator:**
```python
@specialist_agent.output_validator
def validate_specialist_output(ctx: RunContext[AppContext], output: FinalTriageResponse) -> FinalTriageResponse:
    if ctx.partial_output:
        return output
    if len(output.customer_reply.strip()) < 10:
        raise ModelRetry("customer_reply is too short.")
    return output
```

**`@pytest.fixture` -- marks a function as a test fixture:**
```python
@pytest.fixture
def ctx(db: MockDB) -> AppContext:
    return AppContext(db=db, user_email="user1@gmail.com")
```

**`@app.get` / `@app.post` -- registers a function as an HTTP route handler in FastAPI:**
```python
@app.post("/triage", response_model=FinalTriageResponse)
async def triage(request: TriageRequest):
    ...
```

### The key insight about decorators

All of these decorators -- `@agent.tool`, `@app.post`, `@pytest.fixture` -- are the same Python mechanism. They all call a function with your function as the argument and register it somewhere. The difference is what that "somewhere" is: the agent's tool registry, the FastAPI router, pytest's fixture registry.

### Interview question: "What does `@dataclass` do?"

> `@dataclass` is a decorator from the standard library that generates boilerplate code for a class. It reads the class body's type-annotated attributes and automatically creates `__init__`, `__repr__`, and `__eq__`. Without it, you write all of that by hand. It also supports default values, frozen instances, and post-init processing. It's not magic -- it literally generates the same code you'd write yourself.

---

## 6. Type Hints

### What are they?

Type hints are annotations that tell Python (and your IDE / type checker) what types variables and function parameters/returns are expected to be. They are **not enforced at runtime by Python itself** -- they are documentation and static analysis hints.

```python
def greet(name: str) -> str:          # takes a str, returns a str
    return f"Hello, {name}"

def process(items: list[str]) -> None:  # takes a list of strings, returns nothing
    for item in items:
        print(item)
```

### Types used in this project

**Basic types:**
```python
user_email: str
requires_human_approval: bool
order_id: str | None   # either a string or None
```

**`str | None` (union type):** the `|` syntax (Python 3.10+) means "either this type or that type." The equivalent older syntax is `Optional[str]` from `typing`.

**Generic types -- `RunContext[AppContext]`:**
```python
def read_database(ctx: RunContext[AppContext]) -> str:
```

`RunContext[AppContext]` is a generic. `RunContext` is the container. `[AppContext]` specifies what type it contains. This tells the IDE: "when you access `ctx.deps`, its type is `AppContext`." Without the generic parameter, `ctx.deps` would be typed as `Any` -- no autocomplete, no type safety.

**`Callable` -- the type of a function:**
```python
on_status: Callable[[str], Awaitable[None]] | None = None
```

- `Callable` -- it's a function
- `[[str]]` -- the function takes one argument of type `str`
- `Awaitable[None]` -- calling it returns something you can `await` (it's `async def`), and the awaited result is `None`
- `| None = None` -- the whole thing is optional

**`AsyncIterator[str]`:**
```python
async def run_triage_stream_events(ctx: AppContext, user_query: str) -> AsyncIterator[str]:
```

Declares that this async generator yields strings.

### Why type hints matter in practice

1. **IDE autocomplete**: without `RunContext[AppContext]`, typing `ctx.deps.` in your editor shows nothing. With the generic, you get full autocomplete for all `AppContext` fields.
2. **Catching bugs before runtime**: a type checker (`mypy`, `pyright`) can tell you "you're passing a `str` where `AppContext` is expected" before you run the code.
3. **Documentation**: type hints tell the next developer what the function expects and returns, without requiring them to read the implementation.

### Interview question: "Does Python enforce type hints at runtime?"

> No. Python ignores type hints at runtime by default. They're static analysis tools. If you write `def f(x: int): pass` and call `f("hello")`, Python doesn't raise an error. Type hints are for your IDE, for `mypy`/`pyright` type checkers, and for developers reading the code. However, Pydantic is an exception -- Pydantic `BaseModel` classes DO validate types at runtime, converting and rejecting values that don't match.

---

## 7. Callables as First-Class Values

### What does "first-class" mean?

In Python, functions are **first-class values** -- you can store them in variables, pass them as arguments, return them from other functions, and put them in data structures. A function is just an object.

```python
def greet(name: str) -> str:
    return f"Hello, {name}"

# Store in a variable
my_func = greet
print(my_func("Eran"))   # "Hello, Eran"

# Pass as an argument
def apply(func, value):
    return func(value)

print(apply(greet, "Eran"))   # "Hello, Eran"

# Put in a dataclass
@dataclass
class Config:
    formatter: Callable[[str], str] = greet
```

### How it's used in this project

The `on_status` callback in `AppContext`:

```python
# config.py
@dataclass
class AppContext:
    db: MockDB
    user_email: str
    on_status: Callable[[str], Awaitable[None]] | None = None
```

The `on_status` field stores a function. The generator in `triage_service.py` creates a function (`emit_status`) and stores it in the context. Agent tools in `agents.py` call that function without knowing anything about where the events go.

```python
# triage_service.py -- create the function and store it in context
async def emit_status(message: str):
    await status_queue.put(f"data: {json.dumps({'status': message})}\n\n")

stream_ctx = AppContext(db=ctx.db, user_email=ctx.user_email, on_status=emit_status)

# agents.py -- call the function (doesn't know what it does or who created it)
if ctx.deps.on_status:
    await ctx.deps.on_status("Classifying request...")
```

This is the **callback pattern**: Place B (the generator) creates a function and passes it to Place A (the tools) via a shared context. When Place A wants to send information to Place B, it calls that function. Place A doesn't know where the data goes -- that's Place B's business.

### The `if ctx.deps.on_status:` guard

Before calling the callback, you check if it's `None`. The default value is `None` -- when running from `main.py` or tests, nobody sets `on_status`. The `if` guard means: "only call this if someone actually provided a callback." This makes the callback optional without changing any existing code.

### Dynamic function selection with a variable

```python
# main.py
triage_func = run_triage_streaming if IS_STREAM_RESPONSE_OUTPUT else run_triage

results = await asyncio.gather(
    triage_func(AppContext(...), user1["query"]),
    triage_func(AppContext(...), user2["query"]),
)
```

`triage_func` holds a reference to whichever function was selected by the config flag. The `asyncio.gather` call doesn't care which one it is -- it just calls whatever's in `triage_func`. This is a clean way to switch behavior without if/else scattered throughout the code.

### Interview question: "What is a callback? Why use one instead of a direct function call?"

> A callback is a function passed to another piece of code that the other piece calls at a specific event. You use callbacks when two parts of the code are too far apart to call each other directly, or when you want to make behavior optional or configurable. In this project, the tools in `agents.py` can't directly yield SSE events -- they aren't generators and don't have access to the HTTP response. The callback bridges this gap: the generator provides a function, the tools call it, and the generator picks up the result via a queue.

---

## 8. Dataclasses

### What is it?

`@dataclass` is a Python standard library decorator that automatically generates common methods for a class based on its annotated attributes. It generates:
- `__init__`: the constructor
- `__repr__`: the string representation (shown in debugger, `print()`)
- `__eq__`: equality comparison (compares field values)

```python
from dataclasses import dataclass

@dataclass
class AppContext:
    db: MockDB
    user_email: str
    on_status: Callable[[str], Awaitable[None]] | None = None
```

This generates:
```python
def __init__(self, db: MockDB, user_email: str, on_status=None):
    self.db = db
    self.user_email = user_email
    self.on_status = on_status
```

You don't write any of that. `@dataclass` does it automatically based on the annotated fields.

### Default values in dataclasses

Fields with defaults (`= None`) must come after fields without defaults. This is the same rule as function parameters.

```python
@dataclass
class AppContext:
    db: MockDB          # required -- no default
    user_email: str     # required -- no default
    on_status: ... | None = None  # optional -- has default
```

### `dataclass` vs `BaseModel`

| | `@dataclass` | `BaseModel` |
|---|---|---|
| Source | Python standard library | Pydantic library |
| Validation | None -- stores whatever you give it | Full validation and type coercion |
| Use case | Internal infrastructure, trusted data | External data (HTTP requests, LLM output, files) |
| Performance | Lightweight | Validation overhead |
| Serialization | Not built-in | `.model_dump()`, `.model_dump_json()` |

`AppContext` uses `@dataclass` because it's always created by your own trusted Python code. There's nothing to validate. `FinalTriageResponse` uses `BaseModel` because it's the output of an LLM -- it comes from an external source that could return anything.

### Interview question: "When would you use `@dataclass` over a regular class with `__init__`?"

> Any time you have a simple data container class where the fields are just stored and accessed. Writing `__init__`, `__repr__`, and `__eq__` for a 3-field class is tedious boilerplate. `@dataclass` eliminates it. The class body becomes a clean spec of what the class holds. I'd use a regular class when I need custom `__init__` logic that's more than just storing fields, or when I want to add methods that compute derived values.

---

## 9. Enums

### What is it?

An `Enum` is a set of named symbolic constants. Instead of using raw strings like `"refund"` everywhere (which can be misspelled), you define a name (`REFUND`) that maps to a value (`"refund"`).

```python
from enum import Enum

class RequestCategory(str, Enum):
    REFUND = "refund"
    TECHNICAL_SUPPORT = "technical_support"
    GENERAL_QUERY = "general_query"
    UNKNOWN = "unknown"
```

Using it:
```python
# Access by name (your Python code uses this)
category = RequestCategory.REFUND
print(category)         # RequestCategory.REFUND
print(category.value)   # "refund"
print(category.name)    # "REFUND"

# Comparison
if output.category == RequestCategory.REFUND:
    requires_human_approval = True
```

### `str, Enum` -- why inherit from both?

`class RequestCategory(str, Enum)` makes the enum values behave like regular strings. This means:
- `RequestCategory.REFUND == "refund"` is `True`
- The enum serializes to `"refund"` in JSON (not `0` or `"RequestCategory.REFUND"`)
- The LLM can produce `"refund"` as a JSON value and Pydantic will coerce it to `RequestCategory.REFUND`

Without the `str` inheritance, enum values are their own type, and JSON serialization requires extra configuration.

### The `UNKNOWN` sentinel

```python
UNKNOWN = "unknown"
```

`UNKNOWN` is a sentinel value -- a special value used to signal "something went wrong." When the orchestrator fails, the fallback response uses `category="unknown"`. The API layer checks for this:

```python
if result.category == "unknown":
    raise HTTPException(status_code=503, ...)
```

Without the `UNKNOWN` member in the enum, the fallback `FinalTriageResponse` constructor would raise a `ValidationError` because `"unknown"` isn't a valid enum value. The `UNKNOWN` member was added specifically to handle this error propagation path.

### Interview question: "Why use an Enum instead of string constants?"

> Three reasons. First, typos: `"refumd"` is a valid Python string but obviously wrong. `RequestCategory.REFUMD` is a `AttributeError` that you catch immediately. Second, autocomplete: IDEs know the valid members of an enum and offer them in autocomplete. Third, semantics: an enum communicates "these are all the valid values" in one place. Anyone reading the code can see the complete set of categories. A string constant file doesn't have that constraint -- you could add any string.

---

## 10. Exception Handling

### The basics

```python
try:
    result = db.get_order_status(order_id)   # might raise KeyError
except KeyError:                              # catch only KeyError
    return "Order ID could not be found."
except Exception as e:                        # catch any other exception
    return f"ERROR: {str(e)}"
```

The `except` clauses are checked in order. Catch specific exceptions first (narrower), generic exceptions last (wider). `Exception` catches almost everything but not `SystemExit`, `KeyboardInterrupt`, or `BaseException`.

### Raising vs. returning errors in tools

```python
# BAD: raises the exception -- orchestrator crashes, entire request fails
@agent.tool
def fetch_order(ctx, order_id: str) -> str:
    return ctx.deps.db.get_order_status(order_id)  # KeyError propagates up

# GOOD: catches and returns a descriptive string
@agent.tool
def fetch_order(ctx, order_id: str) -> str:
    try:
        return ctx.deps.db.get_order_status(order_id)
    except KeyError:
        return "Order ID could not be found."
```

The orchestrator LLM receives the tool's return value as a string. If you return `"Order ID could not be found."`, the LLM reads that message and can reason about it ("the order doesn't exist, I should tell the customer"). If you raise an exception, the exception propagates up through the tool call machinery and crashes the orchestrator run.

### The three-layer error handling pattern

```python
# Layer 1: Tool level -- absorb delegate failures, return error string
@orchestrator_agent.tool
async def classify_request(ctx, customr_message: str) -> str:
    try:
        result = await classifier_agent.run(...)
        return f"Category: {result.output.category.value}"
    except Exception as e:
        return f"ERROR: Classification failed: {str(e)}. Treat as general_query."

# Layer 2: Service level -- absorb orchestrator failures, return fallback response
async def run_triage(ctx, user_query: str) -> FinalTriageResponse:
    try:
        result = await orchestrator_agent.run(...)
        return apply_business_rules(ctx, result.output)
    except Exception as e:
        logger.error(f"Orchestrator failed: {e}")
        return FinalTriageResponse(category="unknown", ...)  # safe fallback

# Layer 3: API level -- detect fallback response, return proper HTTP error code
async def triage(request: TriageRequest):
    result = await run_triage_streaming(ctx, request.query)
    if result.category == "unknown":
        raise HTTPException(status_code=503, ...)
    return result
```

Each layer catches exceptions at its level of concern. Callers don't need to know about lower-level failures -- they get a clean response type at every boundary.

### `raise` vs `raise ModelRetry`

```python
# ModelRetry: tell the agent to try again with a better response
if len(output.customer_reply.strip()) < 10:
    raise ModelRetry("customer_reply is too short. Provide a meaningful response.")

# KeyError: Python built-in, raised when a dict key isn't found
try:
    return ctx.deps.db.get_order_status(order_id)
except KeyError:
    return "Order ID could not be found."
```

`ModelRetry` is a Pydantic AI exception class that the framework catches and uses to re-prompt the LLM. It's not a normal exception -- raising it inside a validator is the designed way to trigger a retry. Regular exceptions (like `KeyError`) should be caught at the tool level and converted to strings.

### Interview question: "What's the difference between `except Exception` and `except BaseException`?"

> `Exception` is the base class for most "normal" exceptions (errors in your code, IOError, KeyError, ValueError, etc.). `BaseException` is higher up and also catches `SystemExit` (when `sys.exit()` is called), `KeyboardInterrupt` (Ctrl+C), and `GeneratorExit`. You almost always want `except Exception` -- catching `KeyboardInterrupt` accidentally means the user can't stop your program with Ctrl+C, which is very unexpected behavior.

---

## 11. f-strings

### What is it?

An f-string is a string literal prefixed with `f` that can embed Python expressions directly inside `{}` braces. The expressions are evaluated at runtime when the string is created.

```python
name = "Eran"
score = 42

# Old way (% formatting)
msg = "Hello %s, your score is %d" % (name, score)

# Old way (str.format)
msg = "Hello {}, your score is {}".format(name, score)

# f-string (Python 3.6+)
msg = f"Hello {name}, your score is {score}"

# You can put any expression inside {}
msg = f"Score doubled: {score * 2}"
msg = f"Category: {output.category.value}"
msg = f"Email: {ctx.deps.user_email.upper()}"
```

### How it's used in this project

**Structured log messages:**
```python
logger.info(
    f"Orchestrator run complete | user={ctx.user_email} | category={output.category.value} | "
    f"input_tokens={usage.input_tokens} | output_tokens={usage.output_tokens} | requests={usage.requests}"
)
```

Adjacent string literals are automatically concatenated in Python, so two `f"..."` lines on separate lines form one long string. This is more readable than one very long line.

**System prompt injection:**
```python
return f"""
    You are a customer support specialist.
    The customer's email is: {ctx.deps.user_email}
    YOU MUST ALWAYS check the database before responding.
"""
```

Triple-quoted f-strings (`f"""..."""`) support multi-line strings and embedded expressions.

**SSE event formatting:**
```python
yield f"data: {json.dumps({'customer_reply': partial_output.customer_reply})}\n\n"
```

You can call functions and use dictionary literals inside `{}`.

### Interview question: "What are f-strings and when did they appear?"

> f-strings (formatted string literals) were introduced in Python 3.6. They let you embed Python expressions directly in string literals by prefixing the string with `f` and putting expressions in `{}`. They're more readable and faster than `str.format()` or `%` formatting. The expressions are evaluated lazily at the point where the string is used -- they're not pre-evaluated.

---

## 12. Optional Values and None

### The `None` value

`None` is Python's null value. It means "no value here." Every variable can potentially be `None` unless you explicitly prevent it.

```python
order_id: str | None = None   # might be a string or might be nothing
```

### `is None` vs `== None`

```python
# CORRECT: use `is` for None checks
if output.order_id is None:
    requires_human_approval = False

if ctx.deps.on_status is not None:
    await ctx.deps.on_status("Classifying...")

# Also acceptable (but slightly less idiomatic):
if output.order_id:           # True if order_id is a non-empty string
    ...

# AVOID: `== None` works but is not idiomatic Python
if output.order_id == None:   # works, but `is None` is preferred
    ...
```

`is None` checks identity -- "is this the exact `None` object?" `== None` checks equality -- "does this equal `None`?" Since there is only one `None`, they behave the same, but `is None` is the Python convention and is slightly faster.

### The guard pattern

```python
# Check before using -- prevents AttributeError on None
if ctx.deps.on_status:
    await ctx.deps.on_status("Classifying request...")
```

A `None` value is falsy in Python, so `if ctx.deps.on_status:` is equivalent to `if ctx.deps.on_status is not None:` here. If `on_status` is `None`, the `if` body is skipped entirely.

### Using `None` as a sentinel

```python
# FinalTriageResponse fallback response
return FinalTriageResponse(
    order_id=None,   # None signals "no order involved in this failure"
    category="unknown",
    ...
)
```

`None` as a sentinel: a specific value used to signal a special state ("no order ID was provided", "this is a fallback response"). The code downstream checks for `None` to decide how to behave:

```python
if output.order_id is None:
    requires_human_approval = False   # no order = nothing to approve
```

### Interview question: "What is the difference between `None`, `False`, `0`, and `""` in Python?"

> They're all falsy, meaning they evaluate to `False` in a boolean context. But they mean different things. `None` means "no value present." `False` means "this boolean is false." `0` means "this number is zero." `""` means "this string is empty." You shouldn't use them interchangeably. `is None` specifically checks for the absence of a value. Using `""` as a sentinel for "no value" is common but ambiguous -- is it "no value was provided" or "the value is an empty string"? `None` removes that ambiguity.

---

## 13. Python Logging Module

### What is it?

Python's `logging` module is the standard way to emit diagnostic output from a program. Unlike `print()`, it supports:
- Severity levels (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`)
- Named loggers per module
- Configurable output formats
- Routing to files, streams, external services
- Runtime filtering without code changes

### How it's used in this project

**`logger.py` -- the shared factory:**
```python
import logging
import sys

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
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

**Usage in every service file:**
```python
# triage_service.py
logger = get_logger(__name__)   # __name__ is "src.triage_service"

logger.info(
    f"Orchestrator run complete | user={ctx.user_email} | category={output.category.value} | "
    f"input_tokens={usage.input_tokens}"
)

logger.error(f"Orchestrator failed: {e}")
```

**Output format:**
```
2026-02-14 10:06:21 - src.triage_service - INFO - Orchestrator run complete | user=user1@gmail.com | category=refund | input_tokens=543
```

### Why `if not logger.handlers:`?

Python's logging module uses a global registry of named loggers. `logging.getLogger("src.triage_service")` always returns the same logger object for that name. If a module is imported multiple times or `get_logger` is called twice, without the guard you'd add duplicate handlers -- every log message would appear twice (or more). The `if not logger.handlers:` guard ensures the handler is only added once.

### Why f-strings instead of `extra={}`?

```python
# WRONG for terminal output: extra={} is ignored by standard formatters
logger.info("Run complete", extra={"user": "user1@gmail.com", "tokens": 543})
# Output: "2026-02-14 - INFO - Run complete" (the extra fields are silently dropped)

# CORRECT: embed data in the message string
logger.info(f"Run complete | user=user1@gmail.com | tokens=543")
# Output: "2026-02-14 - INFO - Run complete | user=user1@gmail.com | tokens=543"
```

`extra={}` requires a custom formatter that explicitly references those fields in the format string. The standard formatter doesn't know about custom fields, so it ignores them. For simple terminal logging, embed the data directly in the message string using f-strings.

### Severity levels

```python
logger.debug("Detailed trace -- disabled in production")
logger.info("Normal operation -- request received, completed")
logger.warning("Unexpected but not fatal -- category is unknown, returning 503")
logger.error("Something failed -- orchestrator raised an exception")
logger.critical("System is failing -- used sparingly for catastrophic issues")
```

Log level filtering: in production, you often set the log level to `WARNING` or `ERROR` to suppress `DEBUG` and `INFO` noise. All calls below the configured level are ignored at zero cost -- no string formatting, no I/O.

### Interview question: "Why use `logging` instead of `print()`?"

> Three reasons. First, severity: logging lets you distinguish between normal status messages and error conditions. You can filter to only see errors without touching the code. Second, context: every log line automatically includes the timestamp, the module name, and the severity level. Third, routing: logging can send messages to files, external log aggregators (Datadog, CloudWatch), or remote services without changing any application code. `print()` always goes to stdout and has no built-in filtering or routing.

---

## 14. Testing with pytest

### What is pytest?

`pytest` is Python's most popular testing framework. You write test functions that start with `test_`, use `assert` statements to check expectations, and run them with `python -m pytest`.

### Running tests

```bash
python -m pytest tests/test_business_rules.py -v   # run one file, verbose
python -m pytest -v                                 # run all tests
```

`-m` runs pytest as a module (ensures the current directory is in `sys.path`, so imports work). `-v` (verbose) prints each test's name and PASS/FAIL instead of just dots.

### Fixtures

A **fixture** is a function decorated with `@pytest.fixture` that sets up and provides data for tests. pytest automatically detects when a test function has a parameter with the same name as a fixture and calls it.

```python
@pytest.fixture
def db() -> MockDB:
    return MockDB()   # fresh DB for each test

@pytest.fixture
def ctx(db: MockDB) -> AppContext:
    return AppContext(db=db, user_email="user1@gmail.com")
    # Note: ctx depends on db. pytest resolves the dependency automatically
```

```python
def test_refund_forces_human_approval(ctx: AppContext):
    # pytest sees the `ctx` parameter, finds the ctx fixture, runs it, passes the result in
    output = FinalTriageResponse(
        requires_human_approval=False,   # LLM said no approval needed
        order_id="#123",
        category=RequestCategory.REFUND,
        suggested_action="Process Refund",
        customer_reply="Your refund is being processed."
    )
    result = apply_business_rules(ctx, output)
    assert result.requires_human_approval is True   # business rule must override to True
```

**Fixture lifecycle**: by default, a new fixture instance is created for each test function. Test A's `MockDB` is different from Test B's `MockDB` -- they can't corrupt each other.

### `pytest.raises()` -- testing that exceptions are raised

```python
with pytest.raises(ModelRetry):
    validate_specialist_output(mock_ctx, output_with_short_reply)
```

The code inside `with pytest.raises(ModelRetry):` is expected to raise `ModelRetry`. If it does, the test passes. If it does NOT raise, the test fails. If it raises a different exception, the test fails with that exception.

### `assert` -- the test check

```python
assert result.requires_human_approval is True
assert result.order_id == "#123"
assert result.category == RequestCategory.REFUND
```

An `assert` evaluates the expression. If it's `True`, nothing happens. If it's `False`, it raises `AssertionError` and the test fails. pytest enhances this with detailed failure messages showing the actual vs expected values.

### What to test and what NOT to test

```
TEST:                               DON'T TEST:
- apply_business_rules()            - LLM outputs (non-deterministic)
- output validators                 - network calls
- schema validation logic           - the LLM API itself
- any pure Python function
```

Test the deterministic parts. The LLM's output varies every run -- you can't reliably assert what it will say. Business rules and validators are pure Python functions with predictable behavior that you CAN verify.

### Interview question: "What is a pytest fixture and why do you use them?"

> A fixture is a function that sets up data or resources that your tests need. Instead of writing `db = MockDB()` inside every test function, you define it once as a fixture and pytest injects it automatically wherever it's needed. Fixtures isolate tests from each other: by default, each test gets its own fixture instance, so Test A modifying the database doesn't affect Test B. Fixtures can also depend on other fixtures -- pytest automatically builds the dependency chain.

---

## 15. MagicMock — Faking Objects in Tests

### What is it?

`MagicMock` is from `unittest.mock` (Python standard library). It creates a fake object that pretends to be anything. Any attribute you access on it returns another `MagicMock`. Any method you call returns a `MagicMock`. It never raises `AttributeError`.

```python
from unittest.mock import MagicMock

mock = MagicMock()
print(mock.anything)          # returns <MagicMock name='mock.anything'>
print(mock.anything.nested)   # returns <MagicMock ...>
mock.any_method("arg")        # returns <MagicMock ...>
print(mock.some_number)       # returns <MagicMock ...> (not a number, but usable)
```

### How it's used in this project

Testing output validators requires a `RunContext[AppContext]` object. Creating a real `RunContext` requires a live agent run. That requires a real LLM connection, which is expensive and non-deterministic. `MagicMock` lets you fake it:

```python
@pytest.fixture
def mock_ctx():
    ctx = MagicMock()
    ctx.partial_output = False   # we DO set specific values we need
    return ctx

def test_specialist_valid_output_passes(mock_ctx: MagicMock):
    output = FinalTriageResponse(...)
    result = validate_specialist_output(mock_ctx, output)
    assert result == output
```

The validator function only accesses `ctx.partial_output`. `MagicMock` provides a real `False` value for that attribute. Everything else the validator might access is auto-generated as a `MagicMock`, which is fine because the validator doesn't access anything else.

### Setting up a specific behavior

```python
# Default: mock_ctx.partial_output = False (final output)
def test_specialist_skips_validation_for_partial_output(mock_ctx):
    mock_ctx.partial_output = True   # override for this test

    output = FinalTriageResponse(
        suggested_action="",   # would normally fail validation
        customer_reply=""      # would normally fail validation
    )
    result = validate_specialist_output(mock_ctx, output)
    assert result == output   # validation was skipped
```

### Interview question: "When do you use MagicMock?"

> When the code you want to test depends on something expensive, non-deterministic, or hard to create -- like a real database connection, a live API call, or a complex framework object like `RunContext`. `MagicMock` lets you replace that dependency with a fake that returns whatever you configure. You only set up the attributes your code actually uses -- everything else auto-responds. This keeps tests fast, free, and deterministic.

---

## 16. The Module System

### What is a module?

A **module** is any `.py` file. When you import it, Python executes the file from top to bottom and makes its contents available as attributes on the module object.

### Import patterns

```python
# Import the whole module
import asyncio
asyncio.run(main())

# Import specific names from a module
from fastapi import FastAPI, HTTPException
from src.config import AppContext, IS_STREAM_RESPONSE_OUTPUT
from src.schemas import FinalTriageResponse, RequestCategory

# Import everything (use sparingly -- pollutes the namespace)
from src.agents import *
```

### `__name__` -- the module's name

Every module has a `__name__` attribute. When a file runs directly (as the entry point), `__name__` is `"__main__"`. When a file is imported, `__name__` is the module's dotted name.

```python
# main.py
if __name__ == "__main__":
    asyncio.run(main())   # only runs when executed directly, not when imported
```

```python
# src/logger.py
def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)  # name will be "src.triage_service", "api", etc.
    ...

# triage_service.py
logger = get_logger(__name__)   # __name__ here is "src.triage_service"
```

Using `__name__` as the logger name gives each module its own named logger in the logging hierarchy. Log messages from `triage_service.py` appear as `src.triage_service` in the output, so you can tell exactly which file produced each log line.

### Package structure

```
src/              ← package (directory with __init__.py)
├── __init__.py   ← makes `src` importable as a package
├── agents.py     ← module: src.agents
├── config.py     ← module: src.config
├── schemas.py    ← module: src.schemas
├── triage_service.py  ← module: src.triage_service
└── db.py         ← module: src.db
```

**Absolute vs relative imports:**
```python
# Absolute: from the project root
from src.config import AppContext    # always works from project root

# Relative: from the current package
from .config import AppContext       # works within the src/ package
```

This project uses absolute imports throughout (`from src.config import ...`), which is clearer and less prone to confusion.

### Interview question: "What is `if __name__ == '__main__':`?"

> It's a guard that makes a script both importable and directly runnable. When Python runs `main.py` directly, `__name__` is `"__main__"`, so the block runs. When another file imports `main.py` (e.g., for testing), `__name__` is `"main"` (not `"__main__"`), so the block is skipped. Without this guard, importing `main.py` would immediately call `asyncio.run(main())` -- which is never what you want in a test or library import.

---

## 17. JSON Serialization

### What is it?

JSON (JavaScript Object Notation) is a text format for representing structured data. Python's `json` module converts between Python objects and JSON strings.

```python
import json

# Python dict → JSON string
data = {"status": "classifying", "user": "user1@gmail.com"}
json_str = json.dumps(data)   # '{"status": "classifying", "user": "user1@gmail.com"}'

# JSON string → Python dict
parsed = json.loads(json_str)  # {"status": "classifying", "user": "user1@gmail.com"}
parsed["status"]               # "classifying"
```

### How it's used in this project

**SSE event formatting:**
```python
# triage_service.py
yield f"data: {json.dumps({'status': message})}\n\n"
yield f"data: {json.dumps({'customer_reply': partial_output.customer_reply})}\n\n"
yield f"data: {json.dumps({'final': output.model_dump()})}\n\n"
```

Server-Sent Events require a specific text format: `data: {payload}\n\n`. The payload is a JSON string. `json.dumps()` converts a Python dict to a JSON string so it can be embedded in the SSE format.

**Parsing SSE events in the UI:**
```python
# ui.py
def parse_sse_event(raw: str) -> dict | None:
    raw = raw.strip()
    if not raw.startswith("data: "):
        return None
    try:
        return json.loads(raw[len("data: "):])   # strip prefix, parse JSON
    except json.JSONDecodeError:
        return None
```

**Pydantic's `model_dump_json()` for tool returns:**
```python
# agents.py -- tool returning structured data as a JSON string
return result.output.model_dump_json()
```

Tools must return strings (the orchestrator LLM reads text). `model_dump_json()` is Pydantic's built-in method to serialize a `BaseModel` instance to a compact JSON string. This is more reliable than `json.dumps(result.output.dict())` because Pydantic handles enum values, `None`, and nested models correctly.

**`model_dump()` vs `model_dump_json()`:**
```python
response = FinalTriageResponse(category=RequestCategory.REFUND, ...)

response.model_dump()
# Returns a Python dict: {"category": "refund", "requires_human_approval": True, ...}

response.model_dump_json()
# Returns a JSON string: '{"category": "refund", "requires_human_approval": true, ...}'
```

Use `model_dump()` when you need a Python dict (e.g., to pass to `json.dumps()`).
Use `model_dump_json()` when you need a JSON string directly (e.g., to return from a tool).

### Interview question: "What is the difference between `json.dumps` and `json.loads`?"

> `json.dumps` (dump to string) converts a Python object (dict, list, str, int, bool, None) to a JSON-formatted string. `json.loads` (load from string) parses a JSON string back into a Python object. The mnemonic: `dumps` = "dump string", `loads` = "load string". There's also `json.dump` and `json.load` for files instead of strings.

---

## Quick Reference: Concepts by File

| File | Key concepts |
|---|---|
| `src/schemas.py` | `BaseModel`, `Field`, `str Enum`, `str \| None`, type hints |
| `src/config.py` | `@dataclass`, `Callable`, `Awaitable`, type hints, module-level constants |
| `src/agents.py` | `@agent.tool`, `@agent.system_prompt`, `@agent.output_validator`, `async def`, `await`, `try/except`, f-strings, decorators |
| `src/triage_service.py` | `async def`, `await`, `async with`, `async for`, `yield`, `asyncio.Queue`, `AsyncIterator`, f-strings, `json.dumps`, logging |
| `src/logger.py` | `logging` module, `logging.getLogger`, handlers, formatters, `if not logger.handlers` guard |
| `api.py` | FastAPI decorators, `async def`, `await`, `HTTPException`, `StreamingResponse`, logging |
| `main.py` | `asyncio.run`, `asyncio.gather`, `if __name__ == "__main__"`, callable as variable |
| `tests/test_business_rules.py` | `pytest`, `@pytest.fixture`, `assert`, fixture chaining |
| `tests/test_validators.py` | `MagicMock`, `pytest.raises`, fixture with override |

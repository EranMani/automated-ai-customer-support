from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from src.db import MockDB
from src.config import AppContext
from src.schemas import TriageRequest, FinalTriageResponse
from src.triage_service import run_triage_streaming, run_triage_stream_events
from src.logger import get_logger

logger = get_logger(__name__)

"""
NOTE: FastAPI() creates the application instance
NOTE: The title and description appear on the auto-generated docs page
NOTE: GET is for retrieving data
NOTE: POST is for sending data to be processed
NOTE: response_model=FinalTriageResponse - tells FastAPI "the response will be a FinalTriageResponse
NOTE: The ASGI server handles concurrency automatically
NOTE: Each await is a yield point where the event loop can do other work.
NOTE: HTTPException is FastAPI's way of returning error responses with specific status codes
NOTE: JSONResponse lets you build custom JSON responses with any status code.
NOTE: Other status code:
        500 -- Internal Server Error (generic "something broke")
        502 -- Bad Gateway (if you're proxying to another service)
        503 -- Service Unavailable (temporary, try again) -- best fit here
"""

app = FastAPI(
    title="Automated AI Customer Support API",
    description="Multi-agent customer support system powered by Pydantic AI"
)

# Created at module level, once, when the server starts. Every request shares this same database instance.
db_instance = MockDB()

# When someone sends a GET request to /health, call this function
# Check if the server is running without triggering any LLM calls
@app.get("/health")
async def health():
    return {"status": "OK"}

# NOTE: POST endpoint because we're sending data (the customer's email and query)
@app.post("/triage", response_model=FinalTriageResponse)
async def triage(request: TriageRequest):
    logger.info("Triage request received", extra={"email": request.email})
    # Create a per-request context with the shared DB and the user's email
    ctx = AppContext(db=db_instance, user_email=request.email)
    # Using await is critical. This is an async function, and run_triage is async
    # While the LLM is processing, the event loop is free to handle other incoming requests
    result = await run_triage_streaming(ctx, request.query)

    # Detect when the LLM returns an unknown category and return a custom error response
    if result.category == "unknown":
        logger.warning("Orchestrator returned unknown category, returning 503", extra={"email": request.email})
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Service temporarily unavailable. Please try again later.",
                "message": result.customer_reply,
                "suggested_action": result.suggested_action,
            }
        )

    logger.info("Triage request complete", extra={"email": request.email, "category": result.category.value})
    # FastAPI takes the FinalTriageResponse object and serializes it to JSON automatically because it's a Pydantic model.
    return result

@app.post("/triage/stream")
async def triage_stream(request: TriageRequest):
    logger.info("Triage request received", extra={"email": request.email})
    ctx = AppContext(db=db_instance, user_email=request.email)
    # StreamingResponse takes an async generator and sends each yielded value to the client as it arrives, instead of buffering everything and sending it at once.
    return StreamingResponse(
        run_triage_stream_events(ctx, request.query),
        media_type="text/event-stream" # tells the client "this is an SSE stream.
    )
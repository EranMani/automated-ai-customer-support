from fastapi import FastAPI
from src.db import MockDB
from src.config import AppContext
from src.schemas import TriageRequest, FinalTriageResponse
from src.triage_service import run_triage_streaming

"""
NOTE: FastAPI() creates the application instance
NOTE: The title and description appear on the auto-generated docs page
NOTE: GET is for retrieving data
NOTE: POST is for sending data to be processed
NOTE: response_model=FinalTriageResponse - tells FastAPI "the response will be a FinalTriageResponse
NOTE: The ASGI server handles concurrency automatically
NOTE: Each await is a yield point where the event loop can do other work.
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
    # Create a per-request context with the shared DB and the user's email
    ctx = AppContext(db=db_instance, user_email=request.email)
    # Using await is critical. This is an async function, and run_triage is async
    # While the LLM is processing, the event loop is free to handle other incoming requests
    result = await run_triage_streaming(ctx, request.query)
    # FastAPI takes the FinalTriageResponse object and serializes it to JSON automatically because it's a Pydantic model.
    return result

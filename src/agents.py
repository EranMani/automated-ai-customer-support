from pydantic import BaseModel
from pydantic_ai import Agent, RunContext, PromptedOutput
from src.config import AppContext, CLASSIFIER_MODEL, CLASSIFIER_SYSTEM_PROMPT, SPECIALIST_MODEL
from src.tools import fetch_user_tier, fetch_order_status
from src.schemas import CustomerRequestResult, FinalTriageResponse

# the agent that classifies the customer request
classifier_agent = Agent(
    model=CLASSIFIER_MODEL,
    output_type=PromptedOutput(CustomerRequestResult),
    system_prompt=CLASSIFIER_SYSTEM_PROMPT
)

# the agent that provides the final response to the customer
specialist_agent = Agent(
    model=SPECIALIST_MODEL,
    output_type=FinalTriageResponse,
    tools=[fetch_user_tier, fetch_order_status],
    deps_type=AppContext
)

@specialist_agent.system_prompt
def read_database(ctx: RunContext[AppContext]) -> str:
    # NOTE: I added the customer email in the context to make the agent aware of the user's email
    return f"""
        You are a customer support specialist. Analyze the customer message category and provide a fitting response.
        The customer's email is: {ctx.deps.user_email}
        YOU MUST ALWAYS check the database before responding to the customer.
        You can find the following information in the database:
        - User tier (use the customer's email: {ctx.deps.user_email} to fetch the user tier)
        - Order status
        
        CRITICAL: When looking up orders, order IDs in the database include the '#' symbol.
        If a customer says "order #123", you MUST use "#123" (with the #) when calling fetch_order_status.
        Do NOT remove the # symbol - it is required for database lookups.
    """




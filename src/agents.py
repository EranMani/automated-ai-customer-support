from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from config import AppContext, CLASSIFIER_MODEL, CLASSIFIER_SYSTEM_PROMPT, SPECIALIST_MODEL
from tools import fetch_user_tier, fetch_order_status
from schemas import CustomerRequestResult, FinalTriageResponse

# the agent that classifies the customer request
classifier_agent = Agent(
    model=CLASSIFIER_MODEL,
    output_type=CustomerRequestResult,
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
    return f"""
        You are a customer support specialist. Analyze the customer message category and provide a fitting response.
        YOU MUST ALWAYS check the database before responding to the customer.
        You can find the following information in the database:
        - User tier
        - Order status
    """




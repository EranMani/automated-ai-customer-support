"""
NOTE: Use ModelRetry when the model output is low quality (vague reply, missing detail)
NOTE: Use deterministic override in application code when the model output violates a business rule	
NOTE: LLM decides what the customer said, what they want, how to reply
NOTE: The code decides whether the data is real, whether the action is allowed, whether approval is needed
NOTE: Use @agent.tool decorator for agent-specific tools
NOTE: For shared utilities used across multiple agents, pass them via the agent constructor's tools list parameter.
NOTE: Combining agent tool with the system prompt and output validator is great to show the agent capability set visible in one place
NOTE: Every agent run has UsageLimits set to prevent runaway costs - a max request count to cap tool-call loops and a token limit as a cost ceiling
"""

from pydantic_ai import Agent, RunContext, PromptedOutput, ModelRetry
from src.config import AppContext, CLASSIFIER_MODEL, CLASSIFIER_SYSTEM_PROMPT, SPECIALIST_MODEL, MAX_RETRIES
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
    deps_type=AppContext,
    retries=MAX_RETRIES
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

@specialist_agent.tool
def fetch_user_tier(ctx: RunContext[AppContext]) -> str:
    """
        Fetch the user tier level from the database context by using the provided email address
        If the user is not found, return "User ID could not be found."
    """
    try:
        return ctx.deps.db.get_user_tier(ctx.deps.user_email)
    except KeyError:
        return "User ID could not be found."

@specialist_agent.tool
def fetch_order_status(ctx: RunContext[AppContext], order_id: str) -> str:
    """
        Fetch the order status from the database context by using the provided order ID
        If the order is not found, return "Order ID could not be found."
    """
    try:
        return ctx.deps.db.get_order_status(order_id)
    except KeyError:
        return "Order ID could not be found."

@specialist_agent.output_validator
def validate_specialist_output(ctx: RunContext[AppContext], output: FinalTriageResponse) -> FinalTriageResponse:
    # Skip validation for partial streaming outputs -- data is still arriving
    # NOTE: When using run_stream(), the validator is called multiple times with partial outputs. We skip validation for these.
    # NOTE: ctx.partial_output is exactly how I know whether the model is still generating or done
    if ctx.partial_output:
        return output

    if not output.customer_reply or len(output.customer_reply.strip()) < 10:
        raise ModelRetry("customer_reply is too short or empty. Provide a meaningful, professional response to the customer.")

    if not output.suggested_action or len(output.suggested_action.strip()) < 5:
        raise ModelRetry("suggested_action is too vague. Describe the specific action to take")

    return output

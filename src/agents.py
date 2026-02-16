"""
NOTE: Use ModelRetry when the model output is low quality (vague reply, missing detail)
NOTE: Use deterministic override in application code when the model output violates a business rule	
NOTE: LLM decides what the customer said, what they want, how to reply
NOTE: The code decides whether the data is real, whether the action is allowed, whether approval is needed
NOTE: Use @agent.tool decorator for agent-specific tools
NOTE: For shared utilities used across multiple agents, pass them via the agent constructor's tools list parameter.
NOTE: Combining agent tool with the system prompt and output validator is great to show the agent capability set visible in one place
NOTE: Every agent run has UsageLimits set to prevent runaway costs - a max request count to cap tool-call loops and a token limit as a cost ceiling
NOTE: usage=ctx.usage on every delegate call. This rolls all token usage up to the parent, so orchestrator_result.usage() gives you the total cost across ALL agents
NOTE: deps=ctx.deps passes the same AppContext to delegates that need it (specialist and escalation need DB access)
NOTE: Tools return strings (model_dump_json()). The delegate returns structured data internally, but the tool serializes it for the orchestrator.
NOTE: The orchestrator agent comes with few things that might bite:
    1. Token costs explode. 
       The orchestrator will make its own requests PLUS trigger the classifier's requests PLUS trigger the specialist's requests
       We can have 6-10 LLM round-trips per customer query. Every round-trip resends the full history. We might need to increase the token limit
    2. The orchestrator might not call tools in the order you expect.
       We have a nice workflow in the system prompt ("ALWAYS classify first, then..."). The LLM might ignore it. It might skip classification and go straight to the specialist
       It might call escalation without calling the specialist first. You can't guarantee tool call order the way you can with Python if/else.
       That's the trade-off of letting the LLM be the brain.
    3. Error propagation gets layered.
       If the specialist agent fails inside its delegate tool, that exception needs to be caught inside the tool function and returned as a string error to the orchestrator
       If you let it propagate, the entire orchestrator run crashes. You need try/except inside each delegate tool.
    4. Debugging is harder
       When something goes wrong in a 3-layer agent chain, you can't just read the output
       Which agent failed? Did the orchestrator misinterpret the specialist's JSON response? Did the classifier return an unexpected category?
       We need a detailed logging inside each delegate tool.
NOTE: tools are just text in, text out from the orchestrator's perspective. The orchestrator LLM doesn't know or care whether the tool ran a delegate agent, queried a database, or failed entirely
      It just reads the returned string and reasons about it. That's why the error messages matter -- they're instructions to the orchestrator about what to do next.
"""

from pydantic_ai import Agent, RunContext, PromptedOutput, ModelRetry
from src.config import AppContext, CLASSIFIER_MODEL, CLASSIFIER_SYSTEM_PROMPT, SPECIALIST_MODEL, MAX_RETRIES
from src.schemas import CustomerRequestResult, FinalTriageResponse, EscalationResponse

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

escalation_agent = Agent(
    model=SPECIALIST_MODEL,
    output_type=EscalationResponse,
    deps_type=AppContext,
    retries=MAX_RETRIES
)

@escalation_agent.system_prompt
def escalation_prompt(ctx: RunContext[AppContext]) -> str:
    return f"""
        You are a senior customer support escalation specialist.
        You handle high-risk cases that require careful attention: refunds, security issues, account modifications.
        The customer's email is: {ctx.deps.user_email}
        
        Assess the severity, determine the responsible department, and write a detailed internal memo
        explaining why this case needs human review and what action you recommend.
        Also provide a professional holding message for the customer.
    """

@escalation_agent.output_validator
def validate_escalation(ctx: RunContext[AppContext], output: EscalationResponse) -> EscalationResponse:
    # Return the current output if the model is still generating its response
    if ctx.partial_output:
        return output

    # Make sure the severity is one of the allowed values
    if output.severity not in ("low", "medium", "high", "critical"):
        raise ModelRetry("severity must be one of: 'low', 'medium', 'high', 'critical'")

    # Make sure the internal memo is detailed enough for the human reviewer
    if len(output.internal_memo.strip()) < 20:
        raise ModelRetry("internal_memo is too short. Provide a detailed explanation for the human reviewer.")
    return output

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


orchestrator_agent = Agent(
    model=SPECIALIST_MODEL,
    output_type=FinalTriageResponse,
    deps_type=AppContext,
    retries=MAX_RETRIES
)

@orchestrator_agent.system_prompt
def orchestrator_prompt(ctx: RunContext[AppContext]) -> str:
    return f"""
        You are the main customer support orchestrator. 
        The customer's email is: {ctx.deps.user_email}
        
        You have three tools at your disposal:
        1. classify_request - Use this FIRST to understand what the customer wants
        2. handle_support_request - Use this for complex requests that need database lookups and detailed handling
        3. escalate_to_manager - Use this for high-risk cases (refunds, security issues)
        
        Your workflow:
        1. ALWAYS classify the request first
        2. For general queries, respond directly without using other tools
        3. For technical support, use handle_support_request
        4. For refunds or security issues, use BOTH handle_support_request AND escalate_to_manager
        
        Combine all information into your final response.
    """

# This tool calls the classifier agent to classify the customer's message and returns the choosen category as text
@orchestrator_agent.tool
async def classify_request(ctx: RunContext[AppContext], customr_message: str) -> str:
    """Classify the customer's message into a category: refund, technical_support, or general_query."""
    # NOTE: Handle any errors that might happen when running the classifier agent
    try:
        result = await classifier_agent.run(
            user_prompt=customr_message,
            usage=ctx.usage
        )

        return f"Category: {result.output.category.value}"
    except Exception as e:
        return f"ERROR: Classification failed: {str(e)}. Treat as general_query."

# This tool calls the specialist agent to handle the customer's support request and returns the response as JSON
@orchestrator_agent.tool
async def handle_support_request(ctx: RunContext[AppContext], customer_message: str) -> str:
    """Handle a complex customer support request with database lookups and detailed analysis."""
    try:
        result = await specialist_agent.run(
            user_prompt=customer_message,
            deps=ctx.deps,
            usage=ctx.usage,
        )
        return result.output.model_dump_json()
    except Exception as e:
        return f"ERROR: Specialist agent failed: {str(e)}. Respond to the customer directly."

# This tool calls the escalation agent to escalate the customer's support request to a human manager and returns the response as JSON
@orchestrator_agent.tool
async def escalate_to_manager(ctx: RunContext[AppContext], customer_message: str, reason: str) -> str:
    """Escalate a high-risk case to a human manager. Use for refunds, security issues, or account modifications."""
    try:
        result = await escalation_agent.run(
            user_prompt=f"Customer message: {customer_message}\nEscalation reason: {reason}",
            deps=ctx.deps,
            usage=ctx.usage,
        )
        return result.output.model_dump_json()
    except Exception as e:
        return f"ERROR: Escalation failed: {str(e)}. Flag for manual review."

# This validator ensures the final response is valid and meets the business requirements
@orchestrator_agent.output_validator
def validate_orchestrator(ctx: RunContext[AppContext], output: FinalTriageResponse) -> FinalTriageResponse:
    if ctx.partial_output:
        return output

    if not output.customer_reply or len(output.customer_reply.strip()) < 10:
        raise ModelRetry("customer_reply is too short. Provide a meaningful response.")

    return output

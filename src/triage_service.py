from src.agents import classifier_agent, specialist_agent
from src.config import AppContext, MAX_RETRIES
from src.db import MockDB
from src.schemas import FinalTriageResponse, RequestCategory, CustomerRequestResult
from pydantic_ai import ModelRequest, UserPromptPart

async def process_refunds(ctx: AppContext, order_id: str) -> str:
    """
        Process the refund for the given order ID
        If the order is found, update ctx.deps.db with the new order status that is changed to 'REFUNDED'
        If the order is not found, return "Order ID could not be found."
    """
    try:
        success = ctx.db.update_order_status(order_id, "REFUNDED")
        if success:
            return f"SUCCESS: Order {order_id} has been refunded."
        return f"FAILURE: Order {order_id} could not be refunded."
    except Exception as e:
        return f"ERROR: Could not process refund. Details: {e}"

async def run_triage(ctx: AppContext, user_query: str) -> FinalTriageResponse:
    message_history = []
    requires_human_approval = False

    classifier_response = await classifier_agent.run(user_prompt=user_query)
    intent = classifier_response.output.category

    """
        TODO: If it's a refund, we might want to inject extra context, but for now,
               let's just let the Specialist handle the logic for ALL complex intents.
    """

    if intent == RequestCategory.GENERAL_QUERY:
        # Simple heuristic: General queries might not need the heavy agent
        return FinalTriageResponse(
            requires_human_approval=False,
            order_id=None,
            suggested_action="Automated FAQ Response",
            customer_reply="Thank you for your inquiry. A support representative will review your message shortly."
        )

    for attempt in range(MAX_RETRIES):
        try:
            specialist_response = await specialist_agent.run(user_prompt=user_query, deps=ctx, message_history=message_history)
            if intent == RequestCategory.REFUND:
                # NOTE: The classifier's intent drives the approval requirement, not the specialist's judgment
                requires_human_approval = True

            
            return FinalTriageResponse(
                requires_human_approval=requires_human_approval,
                order_id=specialist_response.output.order_id,
                suggested_action=specialist_response.output.suggested_action,
                customer_reply=specialist_response.output.customer_reply
            )

        except Exception as e:
            # NOTE we can use the framework built-in error handling to handle this - raise ModelRetry(e)
            print(f"Attempt {attempt+1} failed: {e}")

            # Append a 'fake' user message telling the model it failed
            message_history.append(
                ModelRequest(parts=[UserPromptPart(content=f"System Error: {str(e)}. Please try a different approach.")])
            )

            # If it's the last attempt, return the fallback
            if attempt == MAX_RETRIES - 1:
                return FinalTriageResponse(
                    requires_human_approval=True,
                    order_id=None,
                    suggested_action="Human Intervention Required",
                    customer_reply="I'm sorry, I'm unable to process your request. A support representative will review your message shortly."
                )

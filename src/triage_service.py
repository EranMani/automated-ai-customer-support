from agents import classifier_agent, specialist_agent
from config import AppContext, MAX_RETRIES
from db import MockDB
from schemas import FinalTriageResponse, RequestCategory
from pydantic_ai import ModelRequest, UserPromptPart

async def run_triage(user_query: str, user_email: str) -> FinalTriageResponse:
    db_instance = MockDB()
    ctx = AppContext(db=db_instance)
    message_history = []

    classifier_prompt = f"User Email: {user_email}\nQuery: {user_query}"
    classifier_response = await classifier_agent.run(user_prompt=classifier_prompt)

    intent = classifier_response.output.category

    """
        TODO: If it's a refund, we might want to inject extra context, but for now,
               let's just let the Specialist handle the logic for ALL complex intents.
    """

    if intent == RequestCategory.GENERAL_QUERY:
        # Simple heuristic: General queries might not need the heavy agent
        return FinalTriageResponse(
            requires_human_approval=False,
            suggested_action="Automated FAQ Response",
            customer_reply="Thank you for your inquiry. A support representative will review your message shortly."
        )

    for attempt in range(MAX_RETRIES):
        try:
            specialist_response = await specialist_agent.run(user_prompt=user_query, deps=ctx, message_history=message_history)
            return FinalTriageResponse(
                requires_human_approval=specialist_response.output.requires_human_approval,
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
                    suggested_action="Human Intervention Required",
                    customer_reply="I'm sorry, I'm unable to process your request. A support representative will review your message shortly."
                )

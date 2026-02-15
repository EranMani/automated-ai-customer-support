from src.agents import classifier_agent, specialist_agent
from src.config import AppContext, SPECIALIST_REQUEST_LIMIT, SPECIALIST_TOTAL_TOKENS_LIMIT
from src.schemas import FinalTriageResponse, RequestCategory
from pydantic_ai import UsageLimits

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
    classifier_response = await classifier_agent.run(user_prompt=user_query)
    print(f"[Usage] Classifier agent - {classifier_response.usage()}")
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

    try:
        """
            We use usage limits to prevent the agent from using too many tokens and running into rate limits
            NOTE: request_limit is the amount of round-trips the LLM can do (including tool calls and retries). when it exceeds this, it raises UsageLimitExceeded
            NOTE: total_tokens_limit is the total input + output tokens across the entire run. Prevents runaway costs.
        """
        specialist_response = await specialist_agent.run(
            user_prompt=user_query, 
            deps=ctx,
            usage_limits=UsageLimits(request_limit=SPECIALIST_REQUEST_LIMIT, total_tokens_limit=SPECIALIST_TOTAL_TOKENS_LIMIT)
        )
        requires_human_approval = specialist_response.output.requires_human_approval

        if intent == RequestCategory.REFUND:
            # NOTE: The classifier's intent drives the approval requirement, not the specialist's judgment
            requires_human_approval = True

        # Business rule: verify the order actually exists in our system
        if specialist_response.output.order_id is not None:
            try:
                ctx.db.get_order_status(specialist_response.output.order_id)
            except KeyError:
                requires_human_approval = False

        # Business rule: no order = nothing to approve
        if specialist_response.output.order_id is None:
            requires_human_approval = False
    
        print(f"[Usage] Specialist agent - {specialist_response.usage()}")

        # Return the model response
        return FinalTriageResponse(
            requires_human_approval=requires_human_approval,
            order_id=specialist_response.output.order_id,
            suggested_action=specialist_response.output.suggested_action,
            customer_reply=specialist_response.output.customer_reply
        )

    except Exception as e:
        print(f"Specialist agent failed after retries: {e}")
       
        return FinalTriageResponse(
            requires_human_approval=True,
            order_id=None,
            suggested_action="Human Intervention Required",
            customer_reply="I'm sorry, I'm unable to process your request. A support representative will review your message shortly."
        )

async def run_triage_streaming(ctx: AppContext, user_query: str) -> FinalTriageResponse:
    # Classify (no streaming needed -- it's fast)
    classifier_response = await classifier_agent.run(user_prompt=user_query)
    print(f"[Usage] Classifier agent - {classifier_response.usage()}")
    intent = classifier_response.output.category

    # General queries don't need the specialist
    if intent == RequestCategory.GENERAL_QUERY:
        return FinalTriageResponse(
            requires_human_approval=False,
            order_id=None,
            suggested_action="Automated FAQ Response",
            customer_reply="Thank you for your inquiry. A support representative will review your message shortly."
        )

    # Stream the specialist response
    # NOTE: Focus on streaming the structured output, not the text
    try:
        async with specialist_agent.run_stream(
            user_prompt=user_query,
            deps=ctx,
            usage_limits=UsageLimits(
                request_limit=SPECIALIST_REQUEST_LIMIT,
                total_tokens_limit=SPECIALIST_TOTAL_TOKENS_LIMIT
            )
        ) as result:
            async for partial_output in result.stream_output():
                if partial_output.customer_reply:
                    print(f"\r{partial_output.customer_reply}", end="", flush=True)
            
            print()  # newline after stream completes

            # run stream returns a stream that might still be in progress, so we need to explicitly await the final output.
            output = await result.get_output()
            # NOTE: get_output() is a blocking call that waits for the stream to complete and returns the final output.
            # call usage only after get_output() has been called, otherwise there will be incomplete numbers
            print(f"[Usage] Specialist agent - {result.usage()}")

        # Apply the SAME business rules as run_triage
        requires_human_approval = output.requires_human_approval

        if intent == RequestCategory.REFUND:
            requires_human_approval = True

        # Business rule: verify the order actually exists in the system
        if output.order_id is not None:
            try:
                ctx.db.get_order_status(output.order_id)
            except KeyError:
                requires_human_approval = False

        # Business rule: no order = nothing to approve
        if output.order_id is None:
            requires_human_approval = False

        return FinalTriageResponse(
            requires_human_approval=requires_human_approval,
            order_id=output.order_id,
            suggested_action=output.suggested_action,
            customer_reply=output.customer_reply
        )

    except Exception as e:
        print(f"Specialist agent failed after retries: {e}")
        return FinalTriageResponse(
            requires_human_approval=True,
            order_id=None,
            suggested_action="Human Intervention Required",
            customer_reply="I'm sorry, I'm unable to process your request. A support representative will review your message shortly."
        )

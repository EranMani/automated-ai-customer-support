from src.agents import classifier_agent, specialist_agent, orchestrator_agent
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

def apply_business_rules(ctx: AppContext, output: FinalTriageResponse) -> FinalTriageResponse:
    requires_human_approval = output.requires_human_approval

    # Business rule: refunds always require human approval
    if output.category == RequestCategory.REFUND:
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
        category=output.category,
        suggested_action=output.suggested_action,
        customer_reply=output.customer_reply
    )

async def run_triage(ctx: AppContext, user_query: str) -> FinalTriageResponse:
    try:
        result = await orchestrator_agent.run(
            user_prompt=user_query,
            deps=ctx,
            usage_limits=UsageLimits(
                request_limit=SPECIALIST_REQUEST_LIMIT,
                total_tokens_limit=SPECIALIST_TOTAL_TOKENS_LIMIT
            )
        )
        print(f"[Usage] Total orchestrator run - {result.usage()}")

        # Business rules still applied deterministically
        output = apply_business_rules(ctx, result.output)
        return output

    except Exception as e:
        print(f"Orchestrator failed: {e}")
        return FinalTriageResponse(
            requires_human_approval=True,
            order_id=None,
            category="unknown",
            suggested_action="Human Intervention Required",
            customer_reply="I'm sorry, I'm unable to process your request."
        )

async def run_triage_streaming(ctx: AppContext, user_query: str) -> FinalTriageResponse:
    # Stream the specialist response
    # NOTE: Focus on streaming the structured output, not the text
    try:
        async with orchestrator_agent.run_stream(
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
            print(f"[Usage] Total orchestrator run - {result.usage()}")

        output = apply_business_rules(ctx, output)
        return output
        
    except Exception as e:
        print(f"Orchestrator failed: {e}")
        return FinalTriageResponse(
            requires_human_approval=True,
            order_id=None,
            category="unknown",
            suggested_action="Human Intervention Required",
            customer_reply="I'm sorry, I'm unable to process your request."
        )

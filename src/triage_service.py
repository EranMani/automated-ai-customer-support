import asyncio
from httpx import request
from src.agents import classifier_agent, specialist_agent, orchestrator_agent
from src.config import AppContext, SPECIALIST_REQUEST_LIMIT, SPECIALIST_TOTAL_TOKENS_LIMIT
from src.schemas import FinalTriageResponse, RequestCategory
from pydantic_ai import UsageLimits
from collections.abc import AsyncIterator
from src.logger import get_logger
import json

logger = get_logger(__name__)

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
            return FinalTriageResponse(
                requires_human_approval=False,
                order_id=output.order_id,   # keep it so logs show what was attempted
                category=output.category,
                suggested_action="Order not found. Ask customer to verify order number.",
                customer_reply=f"We couldn't find order {output.order_id} in our system — "
                            "please double-check your order number and contact us if you need help."
            )

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
        
        # Business rules still applied deterministically
        output = apply_business_rules(ctx, result.output)

        usage = result.usage()
        logger.info(
            f"Orchestrator run complete | user={ctx.user_email} | category={output.category.value} | "
            f"input_tokens={usage.input_tokens} | output_tokens={usage.output_tokens} | requests={usage.requests}"
        )

        return output

    except Exception as e:
        logger.error(f"Orchestrator failed: {e}")
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
            usage = result.usage()
            logger.info(
                f"Orchestrator run streaming complete | user={ctx.user_email} | category={output.category.value} | "
                f"input_tokens={usage.input_tokens} | output_tokens={usage.output_tokens} | requests={usage.requests}"
            )

        output = apply_business_rules(ctx, output)
        return output
        
    except Exception as e:
        logger.error(f"Orchestrator failed: {e}")
        return FinalTriageResponse(
            requires_human_approval=True,
            order_id=None,
            category="unknown",
            suggested_action="Human Intervention Required",
            customer_reply="I'm sorry, I'm unable to process your request."
        )

async def run_triage_stream_events(ctx: AppContext, user_query: str) -> AsyncIterator[str]:
    """Yields server sent events (SSE) as the orchestrator streams its response"""
    # Instead of returning one value, it yields multiple values over time. Each yield sends one piece of data to the client.
    # This function never print()s to the terminal for the client - it yields to the HTTP response instead.

    # Create an async queue to bridge between tool callbacks and the SSE generator
    status_queue: asyncio.Queue[str] = asyncio.Queue()

    async def emit_status(message: str):
        await status_queue.put(f"data: {json.dumps({'status': message})}\n\n")

    # Pass the emit_status function as the on_status callable
    stream_ctx = AppContext(db=ctx.db, user_email=ctx.user_email, on_status=emit_status)

    try:
        async with orchestrator_agent.run_stream(
            user_prompt=user_query,
            deps=stream_ctx,
            usage_limits=UsageLimits(
                request_limit=SPECIALIST_REQUEST_LIMIT,
                total_tokens_limit=SPECIALIST_TOTAL_TOKENS_LIMIT
            )
        ) as result:
            # DRAIN: yield any status events that arrived during tool calls
            while not status_queue.empty():
                yield await status_queue.get()

            async for partial_output in result.stream_output():
                # DRAIN: check for new status events between each partial output
                while not status_queue.empty():
                    yield await status_queue.get()
                    
                if partial_output.customer_reply:
                    # Partial event. sent as each token arrives. The client can display these progressively.
                    yield f"data: {json.dumps({"customer_reply": partial_output.customer_reply})}\n\n"

            output = await result.get_output()
            
            usage = result.usage()
            logger.info(
                f"Orchestrator run complete | user={ctx.user_email} | category={output.category.value} | "
                f"input_tokens={usage.input_tokens} | output_tokens={usage.output_tokens} | requests={usage.requests}"
            )
                
        output = apply_business_rules(ctx, output)
        # This is the SSE format. Server-Sent Events have a specific text format: each event starts with data: , followed by the payload, followed by two newlines
        yield f"data: {json.dumps({'final': output.model_dump()})}\n\n"

    except Exception as e:
        logger.error(f"Orchestrator failed: {e}")
        error_response = FinalTriageResponse(
            requires_human_approval=True,
            order_id=None,
            category="unknown",
            suggested_action="Human Intervention Required",
            customer_reply="I'm sorry, I'm unable to process your request."
        )
        # Final event sent once at the end with the complete output including business rules applied. The client knows to replace the partial display with the final result.
        yield f"data: {json.dumps({'final': error_response.model_dump()})}\n\n"
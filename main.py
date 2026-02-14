"""
NOTE: Each agent run gets its own AppContext with the specific user's email
NOTE: All contexts share the same database instance
NOTE: The context is the per-request data, the database is the shared infrastructure (distinction between shared resources and per-request data such as user identity, session info)
NOTE: Use the LLM for what its good at: understanding natural language and generating responses
NOTE: Enforce buisness rules deterministically in Python code, not in the LLM
"""

import asyncio
from src.triage_service import run_triage, process_refunds
from src.db import MockDB
from src.config import AppContext

async def main():
    db_instance = MockDB()

    user1 = {"email": "user1@gmail.com", "query": "I want a refund for order #123"}
    user2 = {"email": "user2@gmail.com", "query": "What are your business hours?"}
    user3 = {"email": "user1@gmail.com", "query": "I want to return order #999"}
    user4 = {"email": "user3@gmail.com", "query": "I want a refund for order #124"}

    requests = [user1, user2, user3, user4]

    results = await asyncio.gather(
        run_triage(AppContext(db=db_instance, user_email=user1["email"]), user1["query"]),
        run_triage(AppContext(db=db_instance, user_email=user2["email"]), user2["query"]),
        run_triage(AppContext(db=db_instance, user_email=user3["email"]), user3["query"]),
        run_triage(AppContext(db=db_instance, user_email=user4["email"]), user4["query"])
    )

    for i, result in enumerate(results):
        print(f"--- Test Case {i} ---")
        print(f"--- System Analyzing Request: '{requests[i]["query"]}' ---")

        #print(f"Suggested Action: {result.suggested_action}")
        print(f"Dear client: : {result.customer_reply}")
        if result.requires_human_approval:
            # Rebuild the context for this specific user
            user_ctx = AppContext(db=db_instance, user_email=requests[i]["email"])
            await refund_human_feedback(user_ctx, result.order_id)
            
        print("--------------------------------")

async def refund_human_feedback(ctx, order_id: str):
    print("STOP: High-stakes action detected.")
    # NOTE use hardcoded order number for now. Add it dynanically later
    choice = input(f"Admin, do you approve the refund for Order {order_id}? (Y/N): ").strip().upper()

    if choice == "Y":
        print(f"✅ Admin approved the refund for Order {order_id}. Requesting execution...")

        log = await process_refunds(ctx, order_id=order_id)
        print(f"Result: {log}")
    else:
        print(f"❌ Admin declined the refund for Order {order_id}. Request cancelled.")


if __name__ == "__main__":
    asyncio.run(main())

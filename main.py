"""
NOTE: Each agent run gets its own AppContext with the specific user's email
NOTE: All contexts share the same database instance
NOTE: The context is the per-request data, the database is the shared infrastructure (distinction between shared resources and per-request data such as user identity, session info)
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

    requests = [user1, user2, user3]

    results = await asyncio.gather(
        run_triage(AppContext(db=db_instance, user_email=user1["email"]), user1["query"]),
        run_triage(AppContext(db=db_instance, user_email=user2["email"]), user2["query"]),
        run_triage(AppContext(db=db_instance, user_email=user3["email"]), user3["query"])
    )

    for i, result in enumerate(results):
        print(f"--- Test Case {i} ---")
        print(f"--- System Analyzing Request: '{requests[i]["query"]}' ---")

        #print(f"Suggested Action: {result.suggested_action}")
        print(f"Dear client: : {result.customer_reply}")

        if result.requires_human_approval:
            # Rebuild the context for this specific user
            user_ctx = AppContext(db=db_instance, user_email=requests[i]["email"])
            await refund_human_feedback(user_ctx)
            
        print("--------------------------------")

async def refund_human_feedback(ctx):
    print("STOP: High-stakes action detected.")
    # NOTE use hardcoded order number for now. Add it dynanically later
    choice = input(f"Admin, do you approve the refund for Order #123? (Y/N): ").strip().upper()

    if choice == "Y":
        print("✅ Admin approved the refund for Order #123. Requesting execution...")

        log = await process_refunds(ctx, order_id="#123")
        print(f"Result: {log}")
    else:
        print("❌ Admin declined the refund for Order #123. Request cancelled.")


if __name__ == "__main__":
    asyncio.run(main())

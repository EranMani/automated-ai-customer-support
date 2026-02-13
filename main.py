import asyncio
from src.triage_service import run_triage, process_refunds
from src.db import MockDB
from src.config import AppContext

async def main():
    db_instance = MockDB()
    ctx = AppContext(db=db_instance)

    user1 = {"email": "user1@gmail.com", "query": "I want a refund for order #123"}
    user2 = {"email": "user2@gmail.com", "query": "What are your business hours?"}
    user3 = {"email": "user1@gmail.com", "query": "I want to return order #999"}

    results = await asyncio.gather(
        run_triage(user1["query"], user1["email"]),
        run_triage(user2["query"], user2["email"]),
        run_triage(user3["query"], user3["email"])
    )

    for i, result in enumerate(results):
        print(f"--- Test Case {i} ---")
        print(f"Requires Human Approval: {result.requires_human_approval}")

        print(f"Suggested Action: {result.suggested_action}")
        print(f"Customer Reply: {result.customer_reply}")

        if result.requires_human_approval:
            print("⚠️  STOP: High-stakes action detected.")
            choice = input(f"Admin, do you approve the refund for Order #123? (Y/N): ").strip().upper()

            if choice == "Y":
                print("✅ Admin approved the refund for Order #123. Requesting execution...")

                log = await process_refunds(ctx, order_id="#123")
                print(f"Result: {log}")
            else:
                print("❌ Admin declined the refund for Order #123. Request cancelled.")

        print("--------------------------------")


if __name__ == "__main__":
    asyncio.run(main())

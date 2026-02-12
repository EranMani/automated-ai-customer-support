import asyncio
from src.triage_service import run_triage

async def main():
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
        print("--------------------------------")


if __name__ == "__main__":
    asyncio.run(main())

from src.config import AppContext
from pydantic_ai import RunContext

def fetch_user_tier(ctx: RunContext[AppContext], email: str) -> str:
    """
        Fetch the user tier level from the database context by using the provided email address
        If the user is not found, return "User ID could not be found."
    """
    try:
        return ctx.deps.db.get_user_tier(email)
    except KeyError:
        return "User ID could not be found."

def fetch_order_status(ctx: RunContext[AppContext], order_id: str) -> str:
    """
        Fetch the order status from the database context by using the provided order ID
        If the order is not found, return "Order ID could not be found."
    """
    try:
        return ctx.deps.db.get_order_status(order_id)
    except KeyError:
        return "Order ID could not be found."
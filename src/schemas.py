"""
This file contains the structured output schemas for the agents
NOTE: Structured output schemas are the contract between the agent and the rest of your system.
NOTE: Every piece of data your application logic needs to act on should be in the output schema, for example the order_id for the refund logic
NOTE: This makes the LLM's reasoning actionable in determinsic code
NOTE: The LLM decides which order needs a refund. The Python code processes the refund. The FinalTriageResponse schema is the bridge between them
"""
from pydantic import BaseModel, Field
from enum import Enum

class RequestCategory(str, Enum):
    """
        The category of the user's request
        NOTE: LLMs serialize string enums into JSON much more reliably than standard integer-based enums
    """
    REFUND = "refund"
    TECHNICAL_SUPPORT = "technical_support"
    GENERAL_QUERY = "general_query"

class CustomerRequestResult(BaseModel):
    """The result of the customer request classification"""
    category: RequestCategory = Field(description="The classified category of the user's request")

class FinalTriageResponse(BaseModel):
    """The final response to the customer"""
    requires_human_approval: bool = Field(
        description="Set to True if the customer's request is related to a refund, return, or modifying account security — regardless of the current order status. Otherwise, set to False."
    ) 
    # NOTE: Model must return the order_id so we can know which order to refund
    # NOTE: I used str | None because not every request involves an order, like general queries
    order_id: str | None = Field(description="The order ID related to the customer's request, if applicable. Must include the '#' symbol, e.g. '#123'.")
    suggested_action: str = Field(description="How should you respond to the customer's request in a professional and helpful manner?")
    customer_reply: str = Field(description="A single concise sentence that you would say to the customer. It MUST be friendly and professional.")

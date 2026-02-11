"""
This file contains the structured output schemas for the agents
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
    requires_human_approval: bool = Field(description="Set to True ONLY if the suggested action involves issuing a financial refund or modifying account security. Otherwise, set to False.")
    suggested_action: str = Field(description="How should you respond to the customer's request in a professional and helpful manner?")
    customer_reply: str = Field(description="A single concise sentence that you would say to the customer. It MUST be friendly and professional.")

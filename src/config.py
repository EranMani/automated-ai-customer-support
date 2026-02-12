"""
    This module contains the configuration required for the application to run 
"""

from dataclasses import dataclass
from src.db import MockDB
from dotenv import load_dotenv

load_dotenv()

@dataclass
class AppContext:
    """Application context that includes a connection to the mock database"""
    db: MockDB

CLASSIFIER_MODEL = "ollama:llama3.2"
SPECIALIST_MODEL = "openai:gpt-5-nano"

CLASSIFIER_SYSTEM_PROMPT = """
    You are a triage expert. Analyze the customer message and categorize it.

    Available categories:
    - "refund": Customer is requesting a refund or return for a product they purchased. You MUST return this category if the user's request is related to a refund or return.
    - "technical_support": Customer needs help with technical issues or account problems  
    - "general_query": Customer has a general question about business hours, policies, or other non-urgent inquiries

    You MUST respond with ONLY valid JSON in this exact format:
    {"category": "refund"}

    Do NOT use tool calls. Do NOT add explanatory text. Return ONLY the JSON object with the "category" field.
"""

MAX_RETRIES = 3
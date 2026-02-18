"""
    This module contains the configuration required for the application to run 
    NOTE: a callable is anything you can call with parentheses (). functions are values. You can store a function in a variable, pass it to another function, put it in a list 
"""

"""
Why do we use the on_status callable to pass string messages between two distant parts of the code
Each tool call this on_status callable, and the generator in run_triage_stream_events picks up those messages and sends them to the client
"""

from dataclasses import dataclass
from typing import Awaitable, Callable
from src.db import MockDB
from dotenv import load_dotenv

load_dotenv()

@dataclass
class AppContext:
    """Application context that includes a connection to the mock database"""
    db: MockDB
    user_email: str
    # an optional async function that accepts a status message string
    # it takes one argument: a str
    # Awaitable[None] - it returns something you can await (because it's async), and it doesn't return a meaningful value (None)
    # on_status is an optional async function that takes a string message.
    on_status: Callable[[str], Awaitable[None]] | None = None # Making it optional with a default of None

CLASSIFIER_MODEL = "ollama:llama3.2"
SPECIALIST_MODEL = "openai:gpt-4.1-mini"

SPECIALIST_REQUEST_LIMIT = 10
SPECIALIST_TOTAL_TOKENS_LIMIT = 4000

IS_STREAM_RESPONSE_OUTPUT = True

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
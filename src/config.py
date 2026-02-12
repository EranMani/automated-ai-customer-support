"""
    This module contains the configuration required for the application to run 
"""

from dataclasses import dataclass
from db import MockDB

@dataclass
class AppContext:
    """Application context that includes a connection to the mock database"""
    db: MockDB

CLASSIFIER_MODEL = "ollama:llama3.2"
SPECIALIST_MODEL = "openai:gpt-5-nano"

CLASSIFIER_SYSTEM_PROMPT = """
    You are a triage expert. Analyze the customer message and categorize it.
"""

MAX_RETRIES = 3
"""
    This module contains the configuration required for the application to run 
"""

from dataclasses import dataclass
from db import MockDB

@dataclass
class AppContext:
    """Application context that includes a connection to the mock database"""
    db: MockDB
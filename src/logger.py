import logging
import sys

"""
NOTE: To create a logger for each file, we do this:
from src.logger import get_logger
logger = get_logger(__name__) # __name__ is the name of the current module
NOTE: Use of structured logging - the log line carries machine-readable fields, not just a human-readable string
"""

def get_logger(name: str) -> logging.Logger:
    """Get a logger with a specific name."""
    logger = logging.getLogger(name)

    # Prevent adding duplicate handlers
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

    return logger
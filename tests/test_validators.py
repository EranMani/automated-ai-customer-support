"""
Tests for output validators in agents.py

These tests verify that validators correctly reject bad LLM output (via ModelRetry)
and pass good output through. No LLM calls -- we construct outputs manually
and call the validator functions directly.
"""

import pytest
# NOTE: A MagicMock is a fake object that pretends to be anything. When you access any attribute on it, it returns another MagicMock instead of crashing
from unittest.mock import MagicMock
from pydantic_ai import ModelRetry
from src.schemas import FinalTriageResponse, RequestCategory, EscalationResponse
from src.agents import validate_specialist_output, validate_escalation, validate_orchestrator

"""
NOTE: Pydantic BaseModel objects support equality comparison by default - two instances are equal if all their field values match
NOTE: the validator is just a regular Python function. The function doesn't know or care who called it
NOTE: pytest.raises() is a way to test that a function raises an exception.
"""

# NOTE: Creates a fake object that looks enough like RunContext for our validators to work
@pytest.fixture
def mock_ctx():
    """Fake RunContext for testing validators outside of a real agent run."""
    ctx = MagicMock()
    # test the final output validation, not the streaming skip
    ctx.partial_output = False
    return ctx

# NOTE: this test verifies that the validator passes when the output is valid
def test_specialist_valid_output_passes(mock_ctx: MagicMock):
    """Good output should pass through the validator unchanged."""
    output = FinalTriageResponse(
        requires_human_approval=False,
        order_id="#123",
        category=RequestCategory.REFUND,
        suggested_action="Process the refund for order #123",
        customer_reply="Your refund for order #123 is being processed. You'll receive confirmation within 3-5 business days."
    )

    result = validate_specialist_output(mock_ctx, output)

    # result == output checks every field. If any field is different, the assertion will fail.
    assert result == output

# NOTE: this test simulating the LLM giving a lazy one-word response and fires the model retry exception
def test_specialist_rejects_short_customer_reply(mock_ctx: MagicMock):
    output = FinalTriageResponse(
        requires_human_approval=False,
        order_id="#123",
        category=RequestCategory.REFUND,
        suggested_action="Process the refund",
        customer_reply="Ok."
    )

    # This is how we test that a function raises an exception
    # I expect the code inside this block to raise a ModelRetry exception. If it does, the test passes. If it doesn't raise, the test fails.
    with pytest.raises(ModelRetry):
        validate_specialist_output(mock_ctx, output)
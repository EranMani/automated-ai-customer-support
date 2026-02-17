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
NOTE: Test one thing at a time. Each test isolates exactly one validation rule. If a test fails, you know exactly which rule broke.
NOTE: Start by testing the happy path first, then one test per rejection rule, each isolating exactly one bad field
NOTE: Fixtures give you a starting point - you can adjust them per test
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

# NOTE: This test checks the other rejection case in the specialist validator - vague suggested_action
def test_specialist_rejects_vague_suggested_action(mock_ctx):
    """suggested_action shorter than 5 characters should trigger ModelRetry."""
    output = FinalTriageResponse(
        requires_human_approval=False,
        order_id="#123",
        category=RequestCategory.REFUND,
        suggested_action="Help",
        customer_reply="We're looking into your refund request for order #123 and will get back to you shortly."
    )
    with pytest.raises(ModelRetry):
        validate_specialist_output(mock_ctx, output)

# NOTE: This test simulates a streaming scenario where the model is still generating tokens
def test_specialist_skips_validation_for_partial_output(mock_ctx):
    """During streaming, partial outputs should skip validation entirely."""
    # set the partial output to True, skipping validation for partial outputs
    mock_ctx.partial_output = True

    output = FinalTriageResponse(
        requires_human_approval=False,
        order_id=None,
        category=RequestCategory.GENERAL_QUERY,
        suggested_action="",
        customer_reply=""
    )
    result = validate_specialist_output(mock_ctx, output)
    assert result == output

# NOTE: This test verifies that the escalation validator passes when the output severity and internal memo are valid
def test_escalation_valid_output_passes(mock_ctx):
    """Good escalation output should pass through unchanged."""
    output = EscalationResponse(
        severity="high",
        department="billing",
        internal_memo="Customer is requesting a refund for order #123. They are a premium member and the order is still in processing.",
        customer_reply="We're reviewing your case and a senior representative will contact you shortly."
    )
    result = validate_escalation(mock_ctx, output)
    assert result == output

# NOTE: This test verifies that the escalation validator rejects invalid severity values
def test_escalation_rejects_invalid_severity(mock_ctx):
    """Severity must be one of: low, medium, high, critical."""
    output = EscalationResponse(
        severity="urgent",
        department="billing",
        internal_memo="Customer is requesting a refund for order #123. Premium member with processing order.",
        customer_reply="We're reviewing your case and will get back to you."
    )
    with pytest.raises(ModelRetry):
        validate_escalation(mock_ctx, output)

# NOTE: This test verifies that the escalation validator rejects internal memos that are too short
def test_escalation_rejects_short_internal_memo(mock_ctx):
    """Internal memo must be at least 20 characters for the human reviewer."""
    output = EscalationResponse(
        severity="high",
        department="billing",
        internal_memo="Refund needed.",
        customer_reply="We're reviewing your case and a representative will contact you shortly."
    )
    with pytest.raises(ModelRetry):
        validate_escalation(mock_ctx, output)

# NOTE: This test verifies that the orchestrator validator passes when the output is valid
def test_orchestrator_valid_output_passes(mock_ctx):
    """Good orchestrator output should pass through unchanged."""
    output = FinalTriageResponse(
        requires_human_approval=False,
        order_id=None,
        category=RequestCategory.GENERAL_QUERY,
        suggested_action="Answer the question",
        customer_reply="Our business hours are Monday through Friday, 9 AM to 5 PM."
    )
    result = validate_orchestrator(mock_ctx, output)
    assert result == output

# NOTE: This test verifies that the orchestrator validator rejects short customer replies
def test_orchestrator_rejects_short_customer_reply(mock_ctx):
    """Orchestrator should reject short customer replies."""
    output = FinalTriageResponse(
        requires_human_approval=False,
        order_id=None,
        category=RequestCategory.GENERAL_QUERY,
        suggested_action="Answer the question",
        customer_reply="Sure."
    )
    with pytest.raises(ModelRetry):
        validate_orchestrator(mock_ctx, output)
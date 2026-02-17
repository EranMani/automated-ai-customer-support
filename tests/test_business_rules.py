import pytest
from src.config import AppContext
from src.db import MockDB
from src.schemas import FinalTriageResponse, RequestCategory
from src.triage_service import apply_business_rules

"""
    AppContext - apply_business_rules(ctx, output) takes a context as its first argument
    MockDB - because AppContext needs a database instance
    FinalTriageResponse, RequestCategory - because i'll construct fake LLM outputs manually to feed into the business rules
    apply_business_rules - the function for testing

    NOTE: A fixture is a function that creates something your tests need. Instead of writing db = MockDB() inside every single test function, 
          you define it once as a fixture, and pytest automatically passes it to any test that asks for it.
          When pytest sees a test function like def test_something(ctx):, it looks for a fixture named ctx, runs it, and passes the result in.
    NOTE: Why ctx takes db as a parameter? The ctx fixture depends on the db fixture. Pytest sees that ctx needs db, so it runs db() first,
          gets the MockDB instance, and passes it into ctx(db). This is a fixture chain -- pytest handles the dependency order automatically.
    NOTE: Pytest creates a new fixture instance for every test function by default. Test A can't accidentally corrupt data for Test B. Each test starts clean.
    NOTE: Pytest discovers test functions by looking for functions that start with test_, The rest of the name describes what's being tested
    NOTE: Good tests don't just verify tricky cases -- they also guard the happy path against future changes
    NOTE: Run this in terminal - python -m pytest tests/test_business_rules.py -v
    NOTE: Good tests come in pairs -- one where the LLM gets it right, one where it gets it wrong, verifying that business rules fix the wrong cases without breaking the correct ones
    NOTE: When a function transforms data, test both what changed and what should NOT have changed
"""


@pytest.fixture
def db() -> MockDB:
    """Fresh MockDB for each test"""
    return MockDB()

@pytest.fixture
def ctx(db: MockDB) -> AppContext:
    """AppContext with a test user"""
    return AppContext(db=db, user_email="user1@gmail.com")

# The parameter ctx: This tells pytest to inject the ctx fixture. You get a fresh AppContext with a MockDB and user_email="user1@gmail.com"
def test_refund_forces_human_approval(ctx: AppContext):
    """Even if the LLM says no approval needed, refunds always require it."""
    # simulating what the LLM would return in a real scenario
    # We don't need an actual LLM to produce this -- we just construct it directly.
    output = FinalTriageResponse(
        requires_human_approval=False,
        order_id="#123",
        category=RequestCategory.REFUND,
        suggested_action="Process Refund",
        customer_reply="Your refund is being processed."
    )

    result = apply_business_rules(ctx, output)

    # After business rules run, requires_human_approval MUST be True. If it's False, the test fails
    assert result.requires_human_approval is True

# NOTE: this is a regression test in case logic changed in apply_business_rules
def test_refund_with_approval_already_true(ctx: AppContext):
    """If the LLM already set approval to True for a refund, it stays True."""
    output = FinalTriageResponse(
        requires_human_approval=True,
        order_id="#123",
        category=RequestCategory.REFUND,
        suggested_action="Process refund",
        customer_reply="Your refund is being processed."
    )
    result = apply_business_rules(ctx, output)
    assert result.requires_human_approval is True

# NOTE: this test verifies that the order of the business rules produces the correct result when two rules conflict
def test_nonexistent_order_blocks_approval(ctx: AppContext):
    """If the order_id doesn't exist in the DB, approval must be False."""
    output = FinalTriageResponse(
        requires_human_approval=True,
        order_id="#999",
        category=RequestCategory.REFUND,
        suggested_action="Process refund",
        customer_reply="We'll process your refund for order #999."
    )
    result = apply_business_rules(ctx, output)
    assert result.requires_human_approval is False

# NOTE: this test verifies that the business rules produce the correct result when the order_id is None
def test_no_order_id_blocks_approval(ctx):
    """If there's no order_id at all, approval must be False."""
    output = FinalTriageResponse(
        requires_human_approval=True,
        order_id=None,
        category=RequestCategory.REFUND,
        suggested_action="Process refund",
        customer_reply="We'll look into your refund."
    )
    result = apply_business_rules(ctx, output)
    assert result.requires_human_approval is False

# NOTE: this test verifies that the business rules set requires_human_approval to False when the category is GENERAL_QUERY
def test_general_query_no_approval(ctx):
    """General queries should not require approval."""
    output = FinalTriageResponse(
        requires_human_approval=False,
        order_id=None,
        category=RequestCategory.GENERAL_QUERY,
        suggested_action="Answer the question",
        customer_reply="Our business hours are 9-5."
    )
    result = apply_business_rules(ctx, output)
    assert result.requires_human_approval is False

# NOTE: this test verifies that the business rules set requires_human_approval to False when the LLM mistakenly sets it to True for a general query with no order
def test_general_query_llm_sets_approval_true(ctx):
    """Even if LLM mistakenly sets approval for a general query with no order, rules override to False."""
    output = FinalTriageResponse(
        requires_human_approval=True,
        order_id=None,
        category=RequestCategory.GENERAL_QUERY,
        suggested_action="Answer the question",
        customer_reply="Our business hours are 9-5."
    )
    result = apply_business_rules(ctx, output)
    assert result.requires_human_approval is False

# NOTE: this test verifies that the business rules preserve the LLM's decision when the category is TECHNICAL_SUPPORT and the order_id exists
def test_tech_support_with_existing_order_preserves_llm_decision(ctx):
    """For non-refund categories with a valid order, the LLM's decision stands."""
    output = FinalTriageResponse(
        requires_human_approval=False,
        order_id="#123",
        category=RequestCategory.TECHNICAL_SUPPORT,
        suggested_action="Help with technical issue",
        customer_reply="Let me help you with that."
    )
    result = apply_business_rules(ctx, output)
    assert result.requires_human_approval is False

# NOTE: this test verifies that the business rules set requires_human_approval to False when the category is TECHNICAL_SUPPORT and the order_id does not exist
def test_tech_support_with_nonexistent_order(ctx):
    """Tech support referencing a non-existent order -- approval blocked."""
    output = FinalTriageResponse(
        requires_human_approval=True,
        order_id="#999",
        category=RequestCategory.TECHNICAL_SUPPORT,
        suggested_action="Investigate order issue",
        customer_reply="I'll look into that order."
    )
    result = apply_business_rules(ctx, output)
    assert result.requires_human_approval is False

# NOTE: this test checks data integrity since apply_business_rules returns a new final triage response object
# The only value that should change is requires_human_approval
def test_other_fields_preserved(ctx):
    """Business rules should only change requires_human_approval, not other fields."""
    output = FinalTriageResponse(
        requires_human_approval=False,
        order_id="#123",
        category=RequestCategory.REFUND,
        suggested_action="Process the refund immediately",
        customer_reply="Your refund for order #123 is confirmed."
    )
    result = apply_business_rules(ctx, output)

    # Approval changed by business rule
    assert result.requires_human_approval is True
    # Everything else preserved exactly
    assert result.order_id == "#123"
    assert result.category == RequestCategory.REFUND
    assert result.suggested_action == "Process the refund immediately"
    assert result.customer_reply == "Your refund for order #123 is confirmed."
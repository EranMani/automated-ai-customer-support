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
    NOTE: Run this in terminal - python -m pytest tests/test_business_rules.py -v
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
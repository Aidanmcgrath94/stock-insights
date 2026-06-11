"""
Shared test fakes.

Two seams are faked across the suite:

1. The OpenAI client — `model_turn` / `tool_call` build fake chat-completion
   responses; `make_agent` returns a real InsightsAgent whose client emits
   them in order. This drives the real loop logic without network access.

2. The agent itself — `fake_agent` returns a canned AgentResult, for tests
   of layers above the agent (orchestrator, API) that don't care how the
   answer was produced.

Market data uses the in-memory MockProvider from app.services.market_data.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.agents.critic import CriticVerdict
from app.agents.insights_agent import AgentResult, InsightsAgent
from app.main import app, get_insights_agent, get_market_data
from app.models.schemas import ToolCallRecord
from app.services.market_data import MockProvider

# ---------------------------------------------------------------------------
# Fake OpenAI responses
# ---------------------------------------------------------------------------


def tool_call(name, arguments, call_id="call_1"):
    """One tool call as the OpenAI SDK shapes it."""
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def model_turn(content=None, tool_calls=None):
    """One fake chat.completions.create response."""
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(total_tokens=100),
    )


def make_agent(*turns):
    """Real InsightsAgent whose model emits the given turns in order.

    The critic is stubbed (always grounded) so loop tests don't have to
    script its extra LLM call; the critic has its own tests.
    """
    agent = InsightsAgent(api_key="test-key")
    agent._client = MagicMock()
    agent._client.chat.completions.create = AsyncMock(side_effect=list(turns))
    agent._critic = MagicMock()
    agent._critic.review = AsyncMock(return_value=CriticVerdict(grounded=True))
    return agent


# ---------------------------------------------------------------------------
# Fake agent (for orchestrator / API tests)
# ---------------------------------------------------------------------------


def fake_agent(answer="Here is your answer.", tool_calls=None):
    agent = MagicMock()
    agent.answer = AsyncMock(
        return_value=AgentResult(answer=answer, tool_calls=tool_calls or [])
    )
    return agent


def quote_call(ticker, ok=True):
    return ToolCallRecord(tool="get_quote", ticker=ticker, ok=ok)


def profile_call(ticker, ok=True):
    return ToolCallRecord(tool="get_company_profile", ticker=ticker, ok=ok)


# ---------------------------------------------------------------------------
# API client fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def api_client():
    """TestClient factory: pass a fake agent, get a wired client.

    Dependency overrides are cleared after each test.
    """

    def build(agent):
        app.dependency_overrides[get_market_data] = lambda: MockProvider()
        app.dependency_overrides[get_insights_agent] = lambda: agent
        return TestClient(app)

    yield build
    app.dependency_overrides.clear()

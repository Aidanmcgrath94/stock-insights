"""
Orchestrator tests: intent derivation (a pure decision function) and the
request lifecycle contract — agent in, structured response out, failures
translated to typed errors.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agents.orchestrator import UpstreamError, derive_intent, run
from app.models.schemas import AskResponse, QueryIntent, ToolCallRecord
from app.services.conversation import ConversationStore
from tests.conftest import fake_agent, profile_call, quote_call

# ---------------------------------------------------------------------------
# derive_intent: classify the request from what the agent actually did
# ---------------------------------------------------------------------------


def test_one_quote_is_single_stock():
    intent, tickers = derive_intent([quote_call("AAPL")])
    assert intent == QueryIntent.single_stock
    assert tickers == ["AAPL"]


def test_two_quotes_is_comparison():
    intent, tickers = derive_intent([quote_call("TSLA"), quote_call("F")])
    assert intent == QueryIntent.stock_comparison
    assert tickers == ["TSLA", "F"]


def test_profile_wins_over_quote_as_company_lookup():
    intent, tickers = derive_intent([profile_call("AAPL"), quote_call("AAPL")])
    assert intent == QueryIntent.company_lookup
    assert tickers == ["AAPL"]  # deduplicated


def test_news_only_still_counts_as_single_stock():
    calls = [ToolCallRecord(tool="get_company_news", ticker="TSLA", ok=True)]
    intent, tickers = derive_intent(calls)
    assert intent == QueryIntent.single_stock
    assert tickers == ["TSLA"]


def test_failed_calls_do_not_count():
    intent, tickers = derive_intent([quote_call("ZZZZQ", ok=False)])
    assert intent == QueryIntent.unknown
    assert tickers == []


def test_no_tool_calls_is_unknown():
    intent, tickers = derive_intent([])
    assert intent == QueryIntent.unknown
    assert tickers == []


# ---------------------------------------------------------------------------
# run: lifecycle contract
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_returns_structured_response_from_agent_result():
    agent = fake_agent(answer="AAPL is up today.", tool_calls=[quote_call("AAPL")])
    market = MagicMock()

    result = await run("How is AAPL doing?", market, agent, ConversationStore())

    assert isinstance(result, AskResponse)
    assert result.query == "How is AAPL doing?"
    assert result.intent == QueryIntent.single_stock
    assert result.tickers == ["AAPL"]
    assert result.answer == "AAPL is up today."
    assert result.tool_calls == [quote_call("AAPL")]
    assert result.conversation_id  # server-issued on first turn


@pytest.mark.asyncio
async def test_passes_query_provider_and_history_to_agent():
    agent = fake_agent()
    market = MagicMock()
    store = ConversationStore()
    store.append("conv1", "How is AAPL doing?", "AAPL is up 1.3%.")

    await run("What about its P/E?", market, agent, store, conversation_id="conv1")

    args = agent.answer.await_args
    assert args.args == ("What about its P/E?", market)
    history = args.kwargs["history"]
    assert [(t.query, t.answer) for t in history] == [
        ("How is AAPL doing?", "AAPL is up 1.3%.")
    ]


@pytest.mark.asyncio
async def test_successful_turn_is_recorded():
    store = ConversationStore()
    agent = fake_agent(answer="AAPL is up today.")

    result = await run("How is AAPL doing?", MagicMock(), agent, store)

    turns = store.history(result.conversation_id)
    assert [(t.query, t.answer) for t in turns] == [
        ("How is AAPL doing?", "AAPL is up today.")
    ]


@pytest.mark.asyncio
async def test_agent_failure_raises_upstream_error_and_records_nothing():
    agent = MagicMock()
    agent.answer = AsyncMock(side_effect=RuntimeError("OpenAI timeout"))
    store = ConversationStore()

    with pytest.raises(UpstreamError):
        await run("How is AAPL doing?", MagicMock(), agent, store, conversation_id="c1")

    assert store.history("c1") == []  # failed turns don't pollute history

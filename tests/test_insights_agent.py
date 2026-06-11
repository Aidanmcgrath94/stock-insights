"""
Agent loop tests: the decisions the loop makes — when to stop, how tool
results and errors flow back to the model, and the runaway guard.

The OpenAI client is faked (see conftest); market data is the in-memory
MockProvider, so the real dispatch and error handling run.
"""

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from app.agents.insights_agent import MAX_TOOL_ROUNDS
from app.services.market_data import MockProvider
from tests.conftest import make_agent, model_turn, tool_call

# ---------------------------------------------------------------------------
# Loop decisions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_answers_directly_when_no_tools_needed():
    agent = make_agent(model_turn(content="I can help with stock questions."))

    result = await agent.answer("What's the weather?", MockProvider())

    assert result.answer == "I can help with stock questions."
    assert result.tool_calls == []


@pytest.mark.asyncio
async def test_fetches_quote_then_answers():
    agent = make_agent(
        model_turn(tool_calls=[tool_call("get_quote", '{"ticker": "aapl"}')]),
        model_turn(content="AAPL is up today."),
    )

    result = await agent.answer("How is AAPL doing?", MockProvider())

    assert result.answer == "AAPL is up today."
    assert len(result.tool_calls) == 1
    record = result.tool_calls[0]
    assert (record.tool, record.ticker, record.ok) == ("get_quote", "AAPL", True)


@pytest.mark.asyncio
async def test_executes_parallel_tool_calls_in_one_round():
    agent = make_agent(
        model_turn(
            tool_calls=[
                tool_call("get_quote", '{"ticker": "TSLA"}', "c1"),
                tool_call("get_quote", '{"ticker": "F"}', "c2"),
            ]
        ),
        model_turn(content="Both are up."),
    )

    result = await agent.answer("Compare TSLA and F", MockProvider())

    assert [c.ticker for c in result.tool_calls] == ["TSLA", "F"]
    assert all(c.ok for c in result.tool_calls)


@pytest.mark.asyncio
async def test_chains_rounds_for_peer_comparison():
    """The loop supports multi-step plans: peers first, then quotes."""
    agent = make_agent(
        model_turn(tool_calls=[tool_call("get_peers", '{"ticker": "AAPL"}')]),
        model_turn(tool_calls=[tool_call("get_quote", '{"ticker": "AAPL"}')]),
        model_turn(content="Done."),
    )

    result = await agent.answer("Compare AAPL to its competitors", MockProvider())

    assert [c.tool for c in result.tool_calls] == ["get_peers", "get_quote"]
    assert all(c.ok for c in result.tool_calls)


@pytest.mark.asyncio
async def test_tool_results_are_fed_back_to_the_model():
    agent = make_agent(
        model_turn(tool_calls=[tool_call("get_quote", '{"ticker": "AAPL"}')]),
        model_turn(content="done"),
    )

    await agent.answer("How is AAPL doing?", MockProvider())

    second_request = agent._client.chat.completions.create.call_args_list[1]
    tool_messages = [
        m
        for m in second_request.kwargs["messages"]
        if isinstance(m, dict) and m.get("role") == "tool"
    ]
    assert len(tool_messages) == 1
    assert tool_messages[0]["tool_call_id"] == "call_1"
    assert "AAPL" in tool_messages[0]["content"]


@pytest.mark.asyncio
async def test_final_answer_is_reviewed_by_critic():
    agent = make_agent(
        model_turn(tool_calls=[tool_call("get_quote", '{"ticker": "AAPL"}')]),
        model_turn(content="AAPL is up today."),
    )

    await agent.answer("How is AAPL doing?", MockProvider())

    agent._critic.review.assert_awaited_once()
    _, answer, tool_results = agent._critic.review.await_args.args
    assert answer == "AAPL is up today."
    assert len(tool_results) == 1  # the quote result was passed for checking


# ---------------------------------------------------------------------------
# Failure paths: every tool error becomes a result, not an exception
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unknown_ticker_becomes_error_result():
    agent = make_agent(
        model_turn(tool_calls=[tool_call("get_quote", '{"ticker": "ZZZZQ"}')]),
        model_turn(content="I couldn't find data for ZZZZQ."),
    )

    result = await agent.answer("How is ZZZZQ doing?", MockProvider())

    assert result.tool_calls[0].ok is False
    # The model still produced an answer — the request did not fail
    assert result.answer


@pytest.mark.asyncio
async def test_provider_timeout_becomes_error_result():
    provider = MagicMock()
    provider.get_quote = AsyncMock(side_effect=httpx.ReadTimeout("timed out"))

    agent = make_agent(
        model_turn(tool_calls=[tool_call("get_quote", '{"ticker": "AAPL"}')]),
        model_turn(content="Data is temporarily unavailable."),
    )

    result = await agent.answer("How is AAPL doing?", provider)

    assert result.tool_calls[0].ok is False
    assert result.answer


@pytest.mark.asyncio
async def test_malformed_tool_arguments_become_error_result():
    agent = make_agent(
        model_turn(tool_calls=[tool_call("get_quote", "not json")]),
        model_turn(content="Something went wrong."),
    )

    result = await agent.answer("How is AAPL?", MockProvider())

    assert result.tool_calls[0].ok is False


@pytest.mark.asyncio
async def test_unknown_tool_name_becomes_error_result():
    agent = make_agent(
        model_turn(tool_calls=[tool_call("get_dividends", '{"ticker": "AAPL"}')]),
        model_turn(content="I can't fetch dividends."),
    )

    result = await agent.answer("AAPL dividends?", MockProvider())

    assert result.tool_calls[0].ok is False


@pytest.mark.asyncio
async def test_runaway_loop_is_stopped_by_round_guard():
    endless = model_turn(tool_calls=[tool_call("get_quote", '{"ticker": "AAPL"}')])
    agent = make_agent(*[endless] * (MAX_TOOL_ROUNDS + 1))

    with pytest.raises(RuntimeError):
        await agent.answer("How is AAPL doing?", MockProvider())

    assert agent._client.chat.completions.create.await_count == MAX_TOOL_ROUNDS

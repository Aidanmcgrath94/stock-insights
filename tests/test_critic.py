"""
Answer critic tests: verdict parsing and — most importantly — the fail-open
contract. A broken critic must never break answers.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agents.critic import AnswerCritic
from tests.conftest import model_turn

TOOL_RESULTS = [json.dumps({"ticker": "AAPL", "current_price": 189.5})]


def make_critic(response=None, error=None):
    client = MagicMock()
    if error:
        client.chat.completions.create = AsyncMock(side_effect=error)
    else:
        client.chat.completions.create = AsyncMock(return_value=model_turn(content=response))
    return AnswerCritic(client, "gpt-4o-mini")


@pytest.mark.asyncio
async def test_grounded_verdict_is_parsed():
    critic = make_critic(response='{"grounded": true, "issues": []}')

    verdict = await critic.review("How is AAPL?", "AAPL is at $189.50.", TOOL_RESULTS)

    assert verdict.grounded is True
    assert verdict.issues == []


@pytest.mark.asyncio
async def test_ungrounded_verdict_carries_issues():
    critic = make_critic(
        response='{"grounded": false, "issues": ["$250.00 not in tool data"]}'
    )

    verdict = await critic.review("How is AAPL?", "AAPL is at $250.00.", TOOL_RESULTS)

    assert verdict.grounded is False
    assert verdict.issues == ["$250.00 not in tool data"]


@pytest.mark.asyncio
async def test_skips_llm_call_when_no_tool_data():
    """Off-topic answers have nothing to check — no extra cost."""
    critic = make_critic()

    verdict = await critic.review("What's the weather?", "I help with stocks.", [])

    assert verdict.grounded is True
    critic._client.chat.completions.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_malformed_verdict_fails_open():
    critic = make_critic(response="not json at all")

    verdict = await critic.review("q", "answer", TOOL_RESULTS)

    assert verdict.grounded is True


@pytest.mark.asyncio
async def test_llm_failure_fails_open():
    critic = make_critic(error=RuntimeError("OpenAI down"))

    verdict = await critic.review("q", "answer", TOOL_RESULTS)

    assert verdict.grounded is True

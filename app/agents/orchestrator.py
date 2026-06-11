"""
Orchestrator: request lifecycle around the insights agent.

Runs the agent, derives response metadata (intent, tickers) from the tool
calls the agent actually made, and translates failures into clean exceptions.
"""

from __future__ import annotations

import logging

from app.agents.insights_agent import InsightsAgent
from app.models.schemas import AskResponse, QueryIntent, ToolCallRecord
from app.services.conversation import ConversationStore
from app.services.market_data import MarketDataProvider

logger = logging.getLogger(__name__)


class OrchestratorError(Exception):
    """Raised when a request cannot be completed due to its input.
    Maps to a 4xx response."""


class UpstreamError(OrchestratorError):
    """Raised when an external service (Finnhub, OpenAI) fails.
    Not the caller's fault. Maps to a 502 response."""


def derive_intent(calls: list[ToolCallRecord]) -> tuple[QueryIntent, list[str]]:
    """
    Classify the request from what the agent actually did, rather than
    guessing up front. Tickers preserve first-use order, deduplicated.
    """
    tickers: list[str] = []
    for call in calls:
        if call.ok and call.ticker not in tickers:
            tickers.append(call.ticker)

    fetched_profile = any(c.ok and c.tool == "get_company_profile" for c in calls)
    quote_tickers = {c.ticker for c in calls if c.ok and c.tool == "get_quote"}

    if fetched_profile:
        return QueryIntent.company_lookup, tickers
    if len(quote_tickers) >= 2:
        return QueryIntent.stock_comparison, tickers
    if tickers:
        # Quotes, news, financials, or peers for one company
        return QueryIntent.single_stock, tickers
    return QueryIntent.unknown, tickers


async def run(
    query: str,
    market_data: MarketDataProvider,
    agent: InsightsAgent,
    conversations: ConversationStore,
    conversation_id: str | None = None,
) -> AskResponse:
    """
    End-to-end pipeline:
      1. Load prior turns for the conversation (new ID if none given)
      2. Run the tool-calling agent against the query + history
      3. Derive intent/tickers from the tools it used
      4. Record the turn and return a structured :class:`AskResponse`
    """
    conversation_id = conversation_id or conversations.new_id()
    history = conversations.history(conversation_id)
    logger.info(
        'query received: "%s" (conv=%s, %d prior turn(s))',
        query,
        conversation_id,
        len(history),
    )

    try:
        result = await agent.answer(query, market_data, history=history)
    except Exception as exc:
        logger.exception("agent run failed")
        raise UpstreamError("Answer generation is unavailable") from exc

    intent, tickers = derive_intent(result.tool_calls)
    logger.info(
        "intent=%s tickers=%s after %d tool call(s), %d char answer",
        intent.value,
        tickers,
        len(result.tool_calls),
        len(result.answer),
    )

    # Record only successful turns; a failed request shouldn't pollute history
    conversations.append(conversation_id, query, result.answer)

    return AskResponse(
        query=query,
        intent=intent,
        tickers=tickers,
        answer=result.answer,
        tool_calls=result.tool_calls,
        conversation_id=conversation_id,
    )

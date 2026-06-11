"""
News summarizer tests: the summary path and — more importantly — graceful
degradation when the LLM call fails.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agents.news_summarizer import NewsSummarizer
from app.models.schemas import NewsArticle
from tests.conftest import model_turn

ARTICLES = [
    NewsArticle(headline="Apple beats earnings", source="Reuters", date="2026-06-10"),
    NewsArticle(headline="iPhone sales strong", source="WSJ", date="2026-06-09"),
    NewsArticle(headline="Services revenue grows", source="FT", date="2026-06-08"),
]


def make_summarizer(response=None, error=None):
    client = MagicMock()
    if error:
        client.chat.completions.create = AsyncMock(side_effect=error)
    else:
        client.chat.completions.create = AsyncMock(return_value=model_turn(content=response))
    return NewsSummarizer(client, "gpt-4o-mini")


@pytest.mark.asyncio
async def test_returns_llm_summary():
    summarizer = make_summarizer(response="Apple had a strong week.")
    assert await summarizer.summarize("AAPL", ARTICLES) == "Apple had a strong week."


@pytest.mark.asyncio
async def test_llm_failure_degrades_to_headlines():
    summarizer = make_summarizer(error=RuntimeError("OpenAI down"))

    summary = await summarizer.summarize("AAPL", ARTICLES)

    # The tool still produces usable content from the headlines
    assert "Apple beats earnings" in summary


@pytest.mark.asyncio
async def test_empty_llm_response_degrades_to_headlines():
    summarizer = make_summarizer(response="")

    summary = await summarizer.summarize("AAPL", ARTICLES)

    assert "Apple beats earnings" in summary

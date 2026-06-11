"""
News summarizer subagent.

Company news feeds are noisy (up to 8 articles with headlines and blurbs).
Feeding them raw into the main agent's context wastes tokens and dilutes its
attention. This subagent condenses them into 2-3 sentences first.

If the summarization call fails, it degrades gracefully to the top headlines
rather than failing the whole tool call.
"""

from __future__ import annotations

import logging

from openai import AsyncOpenAI

from app.models.schemas import NewsArticle

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You summarize stock market news. Given recent headlines about a company, "
    "produce a 2-3 sentence plain-text summary of the key themes. "
    "Mention only what the headlines support; do not speculate."
)


def _fallback(articles: list[NewsArticle]) -> str:
    """Non-LLM degradation: just the top three headlines."""
    top = "; ".join(a.headline for a in articles[:3])
    return f"Recent headlines: {top}"


class NewsSummarizer:
    """Shares the main agent's OpenAI client; uses a single cheap call."""

    def __init__(self, client: AsyncOpenAI, model: str) -> None:
        self._client = client
        self._model = model

    async def summarize(self, ticker: str, articles: list[NewsArticle]) -> str:
        lines = [f"Recent news for {ticker}:"]
        for a in articles:
            line = f"- [{a.date}] {a.headline} ({a.source})"
            if a.summary:
                line += f" — {a.summary}"
            lines.append(line)

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": "\n".join(lines)},
                ],
                max_tokens=150,
                temperature=0.2,
            )
            summary = (response.choices[0].message.content or "").strip()
            usage = response.usage
            logger.info(
                "summarized %d article(s) for %s, tokens=%s",
                len(articles),
                ticker,
                usage.total_tokens if usage else "?",
            )
            return summary or _fallback(articles)
        except Exception as exc:
            logger.warning("news summarizer failed, using headlines: %s", exc)
            return _fallback(articles)

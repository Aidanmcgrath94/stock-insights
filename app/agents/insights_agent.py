"""
Insights agent: a genuine tool-calling LLM loop.

The model is given five tools (quotes, profiles, peers, news, financials)
and decides for itself which to call, with what tickers, and when it has
enough data to answer. The loop:

    1. Send conversation + tool definitions to the model
    2. If the model requests tool calls, execute them against the market data
       provider and append the results (errors included) to the conversation
    3. Repeat until the model produces a final text answer

Tool failures are returned to the model as results rather than raised, so it
can respond helpfully ("I couldn't find data for XYZ...").
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel

from app.agents.critic import AnswerCritic
from app.agents.news_summarizer import NewsSummarizer
from app.models.schemas import ToolCallRecord
from app.services.conversation import Turn
from app.services.market_data import MarketDataProvider

logger = logging.getLogger(__name__)

# Safety guard: max model round-trips per query (each may carry several
# tool calls). Simple queries finish in 2; peer comparisons need 3
# (peers -> quotes -> answer).
MAX_TOOL_ROUNDS = 5

_SYSTEM_PROMPT = (
    "You are a concise stock market analyst assistant. "
    "Use the provided tools to fetch live market data before answering; "
    "never invent prices. Resolve company names to ticker symbols yourself "
    "(e.g. 'Apple' is AAPL). "
    "For comparisons against unnamed competitors, use get_peers first, then "
    "fetch data for the most relevant peers. "
    "Use get_company_news when asked why a stock is moving or for recent "
    "developments. Use get_basic_financials for valuation questions. "
    "Keep replies to 2-5 sentences of plain text — no markdown, links, "
    "or images. Ground market facts only in fetched data; if some data "
    "could not be fetched, say so plainly rather than guessing. "
    "Use the conversation history to resolve references like 'it' or "
    "'that company', but fetch fresh data rather than reusing numbers "
    "from earlier answers. "
    "If a tool returns an error, briefly explain the problem to the user. "
    "Never give buy, sell, or hold recommendations, price predictions, or "
    "personal investment advice — describe the data neutrally. "
    "Do not add boilerplate disclaimers. "
    "If the question is unrelated to stocks, politely explain what you can "
    "help with instead of calling tools."
)


def _tool(name: str, description: str) -> dict:
    """All tools share the same single-ticker parameter shape."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Exchange ticker symbol, e.g. AAPL",
                    }
                },
                "required": ["ticker"],
            },
        },
    }

TOOLS = [
    _tool(
        "get_quote",
        "Current price quote for a stock: price, daily change, and day range.",
    ),
    _tool(
        "get_company_profile",
        "Company profile: name, industry, market capitalization "
        "(in millions USD), and exchange.",
    ),
    _tool(
        "get_peers",
        "Competitor ticker symbols for a company, for peer comparisons.",
    ),
    _tool(
        "get_company_news",
        "Summary of news about a company from the last 7 days. Use to "
        "explain why a stock is moving or report recent developments.",
    ),
    _tool(
        "get_basic_financials",
        "Valuation metrics: P/E, EPS, 52-week high/low, dividend yield, "
        "net margin, beta.",
    ),
]


class AgentResult(BaseModel):
    answer: str
    tool_calls: list[ToolCallRecord]


class InsightsAgent:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        self._summarizer = NewsSummarizer(self._client, model)
        self._critic = AnswerCritic(self._client, model)

    async def answer(
        self,
        query: str,
        market_data: MarketDataProvider,
        history: list[Turn] | None = None,
    ) -> AgentResult:
        """*history* holds the conversation's prior turns. Only the text is
        replayed — old tool data is never included, so follow-ups get
        referents but prices stay fresh."""
        messages: list = [{"role": "system", "content": _SYSTEM_PROMPT}]
        for turn in history or []:
            messages.append({"role": "user", "content": turn.query})
            messages.append({"role": "assistant", "content": turn.answer})
        messages.append({"role": "user", "content": query})
        records: list[ToolCallRecord] = []
        tool_results: list[str] = []

        for round_num in range(1, MAX_TOOL_ROUNDS + 1):
            start = time.perf_counter()
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=TOOLS,
                max_tokens=300,
                temperature=0.3,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            message = response.choices[0].message
            usage = response.usage

            if not message.tool_calls:
                logger.info(
                    "agent answered after %d round(s), tokens=%s in %.0fms",
                    round_num,
                    usage.total_tokens if usage else "?",
                    elapsed_ms,
                )
                answer = message.content or ""
                # Advisory grounding check; logs internally, never blocks
                await self._critic.review(query, answer, tool_results)
                return AgentResult(answer=answer, tool_calls=records)

            logger.info(
                "round %d: model requested %d tool call(s), tokens=%s in %.0fms",
                round_num,
                len(message.tool_calls),
                usage.total_tokens if usage else "?",
                elapsed_ms,
            )
            messages.append(message)
            # Execute the round's tool calls concurrently; a comparison can
            # request six quotes at once and they're independent
            outcomes = await asyncio.gather(
                *(self._execute(tc, market_data) for tc in message.tool_calls)
            )
            for tool_call, (result, record) in zip(message.tool_calls, outcomes):
                records.append(record)
                tool_results.append(result)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                )

        raise RuntimeError(f"agent exceeded {MAX_TOOL_ROUNDS} tool rounds")

    async def _execute(
        self, tool_call, market_data: MarketDataProvider
    ) -> tuple[str, ToolCallRecord]:
        """Run one tool call; errors become results the model can read."""
        name = tool_call.function.name
        try:
            args = json.loads(tool_call.function.arguments)
            ticker = str(args["ticker"]).upper()
        except (json.JSONDecodeError, KeyError):
            logger.warning("malformed tool arguments: %r", tool_call.function.arguments)
            return (
                json.dumps({"error": "Invalid arguments; expected {\"ticker\": ...}"}),
                ToolCallRecord(tool=name, ticker="?", ok=False),
            )

        try:
            if name == "get_quote":
                result = (await market_data.get_quote(ticker)).model_dump_json()
            elif name == "get_company_profile":
                result = (await market_data.get_company_profile(ticker)).model_dump_json()
            elif name == "get_basic_financials":
                result = (await market_data.get_basic_financials(ticker)).model_dump_json()
            elif name == "get_peers":
                result = json.dumps({"peers": await market_data.get_peers(ticker)})
            elif name == "get_company_news":
                articles = await market_data.get_company_news(ticker)
                summary = await self._summarizer.summarize(ticker, articles)
                result = json.dumps({"news_summary": summary})
            else:
                logger.warning("model requested unknown tool: %s", name)
                return (
                    json.dumps({"error": f"Unknown tool '{name}'"}),
                    ToolCallRecord(tool=name, ticker=ticker, ok=False),
                )
        except ValueError as exc:
            # e.g. unknown ticker — let the model explain it to the user
            logger.warning("tool %s(%s) failed: %s", name, ticker, exc)
            return (
                json.dumps({"error": str(exc)}),
                ToolCallRecord(tool=name, ticker=ticker, ok=False),
            )
        except httpx.HTTPError as exc:
            # Transient provider failure (timeout, 5xx). Feed it back so one
            # slow call doesn't sink an answer built from the others.
            logger.warning("tool %s(%s) provider error: %s", name, ticker, type(exc).__name__)
            return (
                json.dumps({"error": f"Data temporarily unavailable for {ticker}"}),
                ToolCallRecord(tool=name, ticker=ticker, ok=False),
            )

        return result, ToolCallRecord(tool=name, ticker=ticker, ok=True)

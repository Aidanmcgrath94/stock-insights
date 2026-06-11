"""
Market data service with a provider abstraction.

The :class:`MarketDataProvider` protocol defines the interface; the concrete
:class:`FinnhubProvider` implements it using the Finnhub REST API.
An in-memory :class:`MockProvider` is available for testing without network calls.
"""

from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from typing import Protocol

import httpx

from app.models.schemas import (
    BasicFinancials,
    CompanyProfile,
    NewsArticle,
    QuoteData,
)

logger = logging.getLogger(__name__)

MAX_PEERS = 5
MAX_NEWS_ARTICLES = 8
NEWS_LOOKBACK_DAYS = 7


class MarketDataProvider(Protocol):
    """Interface every market-data provider must satisfy."""

    async def get_quote(self, ticker: str) -> QuoteData:
        ...

    async def get_company_profile(self, ticker: str) -> CompanyProfile:
        ...

    async def get_peers(self, ticker: str) -> list[str]:
        ...

    async def get_company_news(self, ticker: str) -> list[NewsArticle]:
        ...

    async def get_basic_financials(self, ticker: str) -> BasicFinancials:
        ...


# ---------------------------------------------------------------------------
# Finnhub implementation
# ---------------------------------------------------------------------------

FINNHUB_BASE = "https://finnhub.io/api/v1"


class FinnhubProvider:
    """Fetches live data from the Finnhub REST API.

    *transport* is an optional httpx transport, used in tests to stub
    responses without network access (e.g. ``httpx.MockTransport``).
    """

    def __init__(
        self, api_key: str, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        self._headers = {"X-Finnhub-Token": api_key}
        self._transport = transport

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=self._transport)

    async def _get(self, path: str, params: dict) -> tuple[object, float]:
        """GET a Finnhub endpoint; returns (parsed JSON, elapsed ms)."""
        start = time.perf_counter()
        async with self._client() as client:
            resp = await client.get(
                f"{FINNHUB_BASE}{path}", params=params, headers=self._headers, timeout=10
            )
            resp.raise_for_status()
            data = resp.json()
        return data, (time.perf_counter() - start) * 1000

    async def get_quote(self, ticker: str) -> QuoteData:
        data, elapsed_ms = await self._get("/quote", {"symbol": ticker})

        if data.get("c") == 0 and data.get("o") == 0:
            logger.warning(
                "finnhub returned no quote data for %s (%.0fms)", ticker, elapsed_ms
            )
            raise ValueError(f"No quote data found for ticker '{ticker}'")

        prev_close = data["pc"]
        current = data["c"]
        change = current - prev_close
        change_pct = (change / prev_close * 100) if prev_close else 0.0

        logger.info(
            "finnhub quote %s $%.2f (%+.2f%%) in %.0fms",
            ticker.upper(),
            current,
            change_pct,
            elapsed_ms,
        )
        return QuoteData(
            ticker=ticker.upper(),
            current_price=current,
            open_price=data["o"],
            high_price=data["h"],
            low_price=data["l"],
            prev_close=prev_close,
            change=round(change, 4),
            change_pct=round(change_pct, 4),
        )

    async def get_company_profile(self, ticker: str) -> CompanyProfile:
        data, elapsed_ms = await self._get("/stock/profile2", {"symbol": ticker})

        if not data or not data.get("name"):
            logger.warning(
                "finnhub returned no profile for %s (%.0fms)", ticker, elapsed_ms
            )
            raise ValueError(f"No company profile found for ticker '{ticker}'")

        logger.info(
            "finnhub profile %s '%s' in %.0fms", ticker.upper(), data["name"], elapsed_ms
        )
        return CompanyProfile(
            ticker=ticker.upper(),
            name=data.get("name", ""),
            industry=data.get("finnhubIndustry", ""),
            market_cap=data.get("marketCapitalization", 0.0),
            exchange=data.get("exchange", ""),
            logo=data.get("logo"),
            weburl=data.get("weburl"),
        )

    async def get_peers(self, ticker: str) -> list[str]:
        data, elapsed_ms = await self._get("/stock/peers", {"symbol": ticker})

        # Finnhub includes the ticker itself in its peer list
        peers = [p for p in data if p.upper() != ticker.upper()][:MAX_PEERS]
        if not peers:
            logger.warning("finnhub returned no peers for %s (%.0fms)", ticker, elapsed_ms)
            raise ValueError(f"No peers found for ticker '{ticker}'")

        logger.info("finnhub peers %s -> %s in %.0fms", ticker.upper(), peers, elapsed_ms)
        return peers

    async def get_company_news(self, ticker: str) -> list[NewsArticle]:
        today = date.today()
        data, elapsed_ms = await self._get(
            "/company-news",
            {
                "symbol": ticker,
                "from": (today - timedelta(days=NEWS_LOOKBACK_DAYS)).isoformat(),
                "to": today.isoformat(),
            },
        )

        articles = [
            NewsArticle(
                headline=item.get("headline", ""),
                source=item.get("source", ""),
                date=date.fromtimestamp(item["datetime"]).isoformat()
                if item.get("datetime")
                else "",
                summary=(item.get("summary") or "")[:300],
            )
            for item in data[:MAX_NEWS_ARTICLES]
            if item.get("headline")
        ]
        if not articles:
            logger.warning("finnhub returned no news for %s (%.0fms)", ticker, elapsed_ms)
            raise ValueError(f"No recent news found for ticker '{ticker}'")

        logger.info(
            "finnhub news %s: %d article(s) in %.0fms", ticker.upper(), len(articles), elapsed_ms
        )
        return articles

    async def get_basic_financials(self, ticker: str) -> BasicFinancials:
        data, elapsed_ms = await self._get(
            "/stock/metric", {"symbol": ticker, "metric": "all"}
        )

        metric = data.get("metric") or {}
        if not metric:
            logger.warning(
                "finnhub returned no financials for %s (%.0fms)", ticker, elapsed_ms
            )
            raise ValueError(f"No financial metrics found for ticker '{ticker}'")

        logger.info("finnhub financials %s in %.0fms", ticker.upper(), elapsed_ms)
        return BasicFinancials(
            ticker=ticker.upper(),
            pe_ttm=metric.get("peTTM") or metric.get("peBasicExclExtraTTM"),
            eps_ttm=metric.get("epsTTM") or metric.get("epsBasicExclExtraItemsTTM"),
            week52_high=metric.get("52WeekHigh"),
            week52_low=metric.get("52WeekLow"),
            dividend_yield=metric.get("currentDividendYieldTTM"),
            net_margin=metric.get("netProfitMarginTTM"),
            beta=metric.get("beta"),
        )


# ---------------------------------------------------------------------------
# Mock provider (for tests and local development without API keys)
# ---------------------------------------------------------------------------

class MockProvider:
    """Returns deterministic, hardcoded data — no network required."""

    _QUOTES: dict[str, QuoteData] = {
        "AAPL": QuoteData(
            ticker="AAPL",
            current_price=189.50,
            open_price=187.00,
            high_price=191.20,
            low_price=186.50,
            prev_close=187.00,
            change=2.50,
            change_pct=1.34,
        ),
        "TSLA": QuoteData(
            ticker="TSLA",
            current_price=245.10,
            open_price=242.00,
            high_price=248.00,
            low_price=240.50,
            prev_close=242.00,
            change=3.10,
            change_pct=1.28,
        ),
        "F": QuoteData(
            ticker="F",
            current_price=12.30,
            open_price=12.10,
            high_price=12.45,
            low_price=12.05,
            prev_close=12.10,
            change=0.20,
            change_pct=1.65,
        ),
    }

    _PROFILES: dict[str, CompanyProfile] = {
        "AAPL": CompanyProfile(
            ticker="AAPL",
            name="Apple Inc.",
            industry="Technology",
            market_cap=2_900_000.0,
            exchange="NASDAQ",
            weburl="https://www.apple.com",
        ),
        "TSLA": CompanyProfile(
            ticker="TSLA",
            name="Tesla Inc.",
            industry="Automobiles",
            market_cap=780_000.0,
            exchange="NASDAQ",
            weburl="https://www.tesla.com",
        ),
        "F": CompanyProfile(
            ticker="F",
            name="Ford Motor Company",
            industry="Automobiles",
            market_cap=48_000.0,
            exchange="NYSE",
            weburl="https://www.ford.com",
        ),
    }

    async def get_quote(self, ticker: str) -> QuoteData:
        ticker = ticker.upper()
        if ticker not in self._QUOTES:
            raise ValueError(f"No mock quote data for '{ticker}'")
        return self._QUOTES[ticker]

    _PEERS: dict[str, list[str]] = {
        "AAPL": ["MSFT", "GOOGL"],
        "TSLA": ["F", "GM"],
        "F": ["GM", "TSLA"],
    }

    async def get_company_profile(self, ticker: str) -> CompanyProfile:
        ticker = ticker.upper()
        if ticker not in self._PROFILES:
            raise ValueError(f"No mock profile data for '{ticker}'")
        return self._PROFILES[ticker]

    async def get_peers(self, ticker: str) -> list[str]:
        ticker = ticker.upper()
        if ticker not in self._PEERS:
            raise ValueError(f"No mock peer data for '{ticker}'")
        return self._PEERS[ticker]

    async def get_company_news(self, ticker: str) -> list[NewsArticle]:
        ticker = ticker.upper()
        if ticker not in self._QUOTES:
            raise ValueError(f"No mock news data for '{ticker}'")
        return [
            NewsArticle(
                headline=f"{ticker} announces quarterly results",
                source="MockWire",
                date="2026-06-10",
                summary=f"{ticker} reported earnings in line with expectations.",
            ),
            NewsArticle(
                headline=f"Analysts weigh in on {ticker} outlook",
                source="MockWire",
                date="2026-06-09",
            ),
        ]

    async def get_basic_financials(self, ticker: str) -> BasicFinancials:
        ticker = ticker.upper()
        if ticker not in self._QUOTES:
            raise ValueError(f"No mock financials data for '{ticker}'")
        return BasicFinancials(
            ticker=ticker,
            pe_ttm=28.5,
            eps_ttm=6.65,
            week52_high=199.62,
            week52_low=164.08,
            dividend_yield=0.5,
            net_margin=25.3,
            beta=1.25,
        )

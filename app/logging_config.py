"""
Logging setup: stdlib logging with a request-scoped correlation ID.

Every log line emitted while handling a request carries the same short
request ID (also returned to the client as the X-Request-ID header), so an
engineer can grep one ID and see the full pipeline for a single request:

    12:01:03 INFO    [a3f8c2d1] app.agents.orchestrator: query received: "Compare TSLA and F"
    12:01:03 INFO    [a3f8c2d1] app.agents.orchestrator: intent=stock_comparison tickers=['TSLA', 'F']
    12:01:03 INFO    [a3f8c2d1] app.services.market_data: finnhub quote TSLA $245.10 (+1.28%) in 142ms
    ...
"""

import logging
import os
import sys
from contextvars import ContextVar

# "-" outside of a request context (e.g. startup, tests)
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

_LOG_FORMAT = "%(asctime)s %(levelname)-7s [%(request_id)s] %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class RequestIdFilter(logging.Filter):
    """Injects the current request ID into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


def setup_logging() -> None:
    """Configure the root logger. Idempotent; level via LOG_LEVEL env var."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    handler.addFilter(RequestIdFilter())

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())

    # httpx logs every request at INFO; our market_data logs already cover
    # those calls with more signal (ticker, price, latency)
    logging.getLogger("httpx").setLevel(logging.WARNING)

"""
LLM answer critic.

After the agent produces its final answer, a second model call reviews it
against the data the tools actually returned and flags claims the data does
not support — wrong numbers, tickers that were never fetched, causal claims
with no basis.

Two hard rules keep it safe and cheap:

- **Advisory**: verdicts are logged, never blocking. The user still gets
  their answer; ungrounded ones are visible (and alertable) in the logs.
- **Fail open**: any critic failure (API error, malformed verdict) counts
  as grounded. A broken critic must never break answers.

Skipped entirely when no tools ran — there is no data to check against.
"""

from __future__ import annotations

import json
import logging

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a strict fact-checker for a stock market assistant. "
    "You receive market data fetched from tools (as JSON) and the "
    "assistant's final answer. Decide whether every factual claim in the "
    "answer is supported by that data. "
    "Tolerate rounding, unit conversions (e.g. millions to billions), and "
    "qualitative phrasing. Flag numbers absent from the data, companies or "
    "tickers that were never fetched, and causal claims the data does not "
    "support. "
    'Respond with JSON only: {"grounded": <bool>, "issues": ["<short reason>", ...]}'
)


class CriticVerdict(BaseModel):
    grounded: bool
    issues: list[str] = Field(default_factory=list)


class AnswerCritic:
    """Shares the main agent's OpenAI client; one cheap call per answer."""

    def __init__(self, client: AsyncOpenAI, model: str) -> None:
        self._client = client
        self._model = model

    async def review(
        self, query: str, answer: str, tool_results: list[str]
    ) -> CriticVerdict:
        if not tool_results:
            # No data was fetched (e.g. off-topic question) — nothing to check
            return CriticVerdict(grounded=True)

        user_message = (
            f"Question: {query}\n\n"
            "Tool data:\n" + "\n".join(tool_results) + "\n\n"
            f"Answer to check:\n{answer}"
        )

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                response_format={"type": "json_object"},
                max_tokens=200,
                temperature=0,
            )
            data = json.loads(response.choices[0].message.content or "")
            verdict = CriticVerdict(
                grounded=bool(data.get("grounded", True)),
                issues=[str(issue) for issue in data.get("issues", [])],
            )
        except Exception as exc:
            logger.warning("critic failed open: %s", exc)
            return CriticVerdict(grounded=True)

        if verdict.grounded:
            logger.info("critic: answer grounded in tool data")
        else:
            logger.warning("critic: answer not grounded: %s", verdict.issues)
        return verdict

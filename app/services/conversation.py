"""
In-memory conversation store.

Deliberately not a database: the app runs as a single worker, history is
only needed so follow-up questions have referents ("what about its P/E?"),
and losing it on restart is acceptable. Two caps bound memory: turns kept
per conversation, and total conversations (oldest evicted first).
"""

from __future__ import annotations

import uuid
from collections import OrderedDict

from pydantic import BaseModel

MAX_TURNS = 6
MAX_CONVERSATIONS = 100


class Turn(BaseModel):
    query: str
    answer: str


class ConversationStore:
    def __init__(self) -> None:
        # OrderedDict so the least-recently-used conversation evicts first
        self._conversations: OrderedDict[str, list[Turn]] = OrderedDict()

    def new_id(self) -> str:
        return uuid.uuid4().hex[:12]

    def history(self, conversation_id: str) -> list[Turn]:
        """Prior turns, oldest first. Unknown IDs are an empty conversation."""
        return list(self._conversations.get(conversation_id, []))

    def append(self, conversation_id: str, query: str, answer: str) -> None:
        turns = self._conversations.setdefault(conversation_id, [])
        turns.append(Turn(query=query, answer=answer))
        del turns[:-MAX_TURNS]  # keep only the most recent turns

        self._conversations.move_to_end(conversation_id)
        while len(self._conversations) > MAX_CONVERSATIONS:
            self._conversations.popitem(last=False)

"""
Conversation store tests: history round-trips and the two memory caps.
"""

from app.services.conversation import MAX_CONVERSATIONS, MAX_TURNS, ConversationStore


def test_history_round_trip():
    store = ConversationStore()
    cid = store.new_id()

    store.append(cid, "How is AAPL doing?", "AAPL is up 1.3%.")
    store.append(cid, "What about its P/E?", "AAPL trades at 28.5x.")

    turns = store.history(cid)
    assert [(t.query, t.answer) for t in turns] == [
        ("How is AAPL doing?", "AAPL is up 1.3%."),
        ("What about its P/E?", "AAPL trades at 28.5x."),
    ]


def test_unknown_id_is_empty_conversation():
    assert ConversationStore().history("nope") == []


def test_turns_capped_to_most_recent():
    store = ConversationStore()
    cid = store.new_id()

    for i in range(MAX_TURNS + 3):
        store.append(cid, f"q{i}", f"a{i}")

    turns = store.history(cid)
    assert len(turns) == MAX_TURNS
    assert turns[0].query == "q3"  # oldest three evicted
    assert turns[-1].query == f"q{MAX_TURNS + 2}"


def test_oldest_conversation_evicted_beyond_cap():
    store = ConversationStore()

    store.append("first", "q", "a")
    for i in range(MAX_CONVERSATIONS):
        store.append(f"conv{i}", "q", "a")

    assert store.history("first") == []  # evicted
    assert store.history(f"conv{MAX_CONVERSATIONS - 1}") != []


def test_active_conversation_survives_eviction():
    """Appending refreshes recency, so a busy conversation is not evicted."""
    store = ConversationStore()
    store.append("busy", "q0", "a0")

    for i in range(MAX_CONVERSATIONS - 1):
        store.append(f"conv{i}", "q", "a")
    store.append("busy", "q1", "a1")  # refresh, then push others past the cap
    for i in range(MAX_CONVERSATIONS - 1, MAX_CONVERSATIONS + 5):
        store.append(f"conv{i}", "q", "a")

    assert len(store.history("busy")) == 2

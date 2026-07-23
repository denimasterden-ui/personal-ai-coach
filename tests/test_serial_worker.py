"""Per-chat serial worker — the fix for the two-voice-notes race (ADR 0002).

Covers TEST_PLAN.md rows: test_serial_worker, cross-chat parallelism.
Guards against the regression where two messages arriving ~20s apart triggered
two concurrent _ask_brain calls on the same in-RAM session.
"""
import asyncio

import pytest

import bot


@pytest.fixture
def fast_debounce(monkeypatch):
    monkeypatch.setattr(bot, "DEBOUNCE_SEC", 0.02)
    bot._chat.clear()
    bot._inflight.clear()


@pytest.fixture
def stub_io(monkeypatch):
    """Stub all outbound I/O; record turn order and peak concurrency of _ask_brain."""
    state = {"concurrent": 0, "peak": 0, "turns": []}

    async def fake_ask(client, tenant, session_id, text, model=None):
        state["concurrent"] += 1
        state["peak"] = max(state["peak"], state["concurrent"])
        state["turns"].append(text)
        await asyncio.sleep(0.05)  # simulate brain work long enough to overlap
        state["concurrent"] -= 1
        return f"answer to: {text[:20]}"

    async def noop(*a, **k):
        return None

    async def keepalive(*a, **k):
        try:
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            raise

    monkeypatch.setattr(bot, "_ask_brain", fake_ask)
    monkeypatch.setattr(bot, "_send_text", noop)
    monkeypatch.setattr(bot, "_tg", noop)
    monkeypatch.setattr(bot, "_typing_keepalive", keepalive)
    monkeypatch.setattr(bot.analytics, "log", lambda *a, **k: None)
    return state


async def _drain():
    """Wait until all worker tasks finish."""
    for _ in range(200):
        if all(t.done() for t in list(bot._inflight)):
            break
        await asyncio.sleep(0.01)


async def test_second_message_serializes_not_races(fast_debounce, stub_io):
    """Message arriving mid-processing must queue as the next turn, never overlap."""
    bot._buffer(None, 111, "first", kind="voice")
    await asyncio.sleep(0.03)          # let worker pick up turn #1 (inside _ask_brain)
    bot._buffer(None, 111, "second", kind="voice")  # arrives during turn #1
    await _drain()

    assert stub_io["peak"] == 1, "two _ask_brain calls overlapped — race is back"
    assert stub_io["turns"] == ["first", "second"], "turns out of order"


async def test_burst_coalesces_into_one_turn(fast_debounce, stub_io):
    """Several messages within the debounce window collapse into a single turn."""
    bot._buffer(None, 222, "a", kind="text")
    bot._buffer(None, 222, "b", kind="text")
    bot._buffer(None, 222, "c", kind="text")
    await _drain()

    assert len(stub_io["turns"]) == 1
    assert stub_io["turns"][0] == "a\n\nb\n\nc"


async def test_cross_chat_runs_in_parallel(fast_debounce, stub_io):
    """Different chats are independent — one chat must not block another."""
    bot._buffer(None, 1, "x", kind="text")
    bot._buffer(None, 2, "y", kind="text")
    await _drain()

    assert stub_io["peak"] == 2, "separate chats should process concurrently"
    assert sorted(stub_io["turns"]) == ["x", "y"]

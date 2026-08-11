"""Продуктовая обратная связь после третьего хода (aicoach-dev#20, ADR 0004).

Бот — не коуч — спрашивает один раз отдельным сообщением, оказалось ли полезным.
Ответ человека — это факт о продукте, а не о человеке: он не уходит в мозг как
ход и не попадает в извлекающий проход. Все тесты идут через шов обвязки
(_handle / _worker) с фейковым Telegram.
"""
import asyncio

import pytest

import bot


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    """Point the product store at a temp file; start each test clean."""
    monkeypatch.setattr(bot, "PRODUCT_STORE_FILE", tmp_path / "contacts.enc")
    bot._contacts.clear()
    bot._feedback_pending.clear()


@pytest.fixture
def tg(monkeypatch):
    """Capture outbound Telegram calls; neuter access gating and analytics."""
    sent = []

    async def fake_tg(client, method, **params):
        sent.append((method, params))
        if method == "getFile":
            return {"result": {"file_path": "voice/clip.oga"}}
        return {"ok": True}

    monkeypatch.setattr(bot, "_tg", fake_tg)
    monkeypatch.setattr(bot, "PUBLIC_MODE", True)
    monkeypatch.setattr(bot, "_rate_limited", lambda _: False)
    monkeypatch.setattr(bot.analytics, "has_seen", lambda _: True)
    monkeypatch.setattr(bot.analytics, "log", lambda *a, **k: None)
    return sent


@pytest.fixture
def buffered(monkeypatch):
    """Capture what reaches the turn buffer — feedback must never land here."""
    out = []
    monkeypatch.setattr(bot, "_buffer", lambda c, cid, text, **kw: out.append(text))
    return out


class FakeClient:
    """Voice download + session-delete stand-in."""

    async def get(self, url, **kw):
        class R:
            content = b"\x00audio"
        return R()

    async def delete(self, url, **kw):
        return None


def _count_questions(sent):
    """How many times the feedback question was sent."""
    return sum(1 for m, p in sent
               if m == "sendMessage" and p.get("text") == bot.FEEDBACK_QUESTION)


def _capture_turn(sink):
    async def _f(client, tenant, session, text, model=None, turn_id=None,
                 light_intro=False):
        sink.append(text)
        return "ответ коуча"
    return _f


async def _noop(*a, **k):
    return None


async def _drain(chat_id, timeout=5.0):
    """Wait until the chat has no buffered parts and no running worker."""
    async def _wait():
        while True:
            st = bot._chat.get(chat_id)
            if st and not st["preparing"] and not st["parts"]:
                w = st["worker"]
                if w is None or w.done():
                    return
            await asyncio.sleep(0.05)
    await asyncio.wait_for(_wait(), timeout=timeout)


async def _run_turns(monkeypatch, chat_id, n):
    """Push n text turns through the harness seam and wait for each to finish."""
    monkeypatch.setattr(bot, "DEBOUNCE_SEC", 0)
    turns = []
    monkeypatch.setattr(bot, "_ask_brain", _capture_turn(turns))
    monkeypatch.setattr(bot, "_send_text", _noop)
    for i in range(n):
        await bot._handle(FakeClient(),
                          {"chat": {"id": chat_id}, "message_id": i + 1,
                           "text": f"ход {i + 1}"})
        await _drain(chat_id)
    return turns


# ── question is asked after the 3rd turn, exactly once ─────────────────────


async def test_no_question_before_third_turn(tg, monkeypatch):
    turns = await _run_turns(monkeypatch, 1, 2)
    assert len(turns) == 2
    assert _count_questions(tg) == 0, "no feedback question before the 3rd turn"


async def test_question_asked_after_third_turn(tg, monkeypatch):
    turns = await _run_turns(monkeypatch, 1, 3)
    assert len(turns) == 3
    assert _count_questions(tg) == 1, "feedback question asked after the 3rd turn"
    assert 1 in bot._feedback_pending


async def test_question_asked_exactly_once(tg, monkeypatch):
    """Even if more turns follow (through _worker directly, bypassing the
    feedback interception in _handle), the question fires only once."""
    turns = await _run_turns(monkeypatch, 1, 3)
    assert _count_questions(tg) == 1

    # Two more turns through _worker — question must not repeat.
    for i in range(2):
        st = bot._state(1)
        st["parts"] = [f"ещё ход {i}"]
        await bot._worker(FakeClient(), 1)

    assert _count_questions(tg) == 1, "feedback question must not be repeated"


# ── answer does not reach the brain ─────────────────────────────────────────


async def test_feedback_answer_does_not_go_to_brain(tg, monkeypatch):
    """The core acceptance test: the answer is captured as product feedback and
    never buffered as a turn — otherwise it enters the brain and the extraction
    pass, and a product opinion settles in memory as a fact about the person."""
    await _run_turns(monkeypatch, 1, 3)
    assert 1 in bot._feedback_pending

    # Spy on _buffer only AFTER the 3 turns, to prove the feedback answer never
    # enters the brain path.
    buf = []
    monkeypatch.setattr(bot, "_buffer", lambda *a, **kw: buf.append(a))

    await bot._handle(FakeClient(),
                      {"chat": {"id": 1}, "message_id": 99,
                       "text": "Очень полезно, спасибо"})

    assert buf == [], "feedback answer must not be buffered for the brain"
    assert bot._contacts["1"]["feedback"] == "Очень полезно, спасибо"
    assert 1 not in bot._feedback_pending
    assert any(p.get("text") == bot.FEEDBACK_THANKS
               for m, p in tg if m == "sendMessage"), "person is thanked"


# ── voice feedback is transcribed and captured ──────────────────────────────


async def test_voice_feedback_is_transcribed_and_captured(tg, monkeypatch):
    await _run_turns(monkeypatch, 42, 3)
    assert 42 in bot._feedback_pending

    async def fake_transcribe(client, audio):
        return "Отлично, прям попало"
    monkeypatch.setattr(bot, "_transcribe", fake_transcribe)

    await bot._handle(FakeClient(),
                      {"chat": {"id": 42}, "message_id": 1,
                       "voice": {"file_id": "v1"}})

    assert bot._contacts["42"]["feedback"] == "Отлично, прям попало"
    assert 42 not in bot._feedback_pending
    assert any(p.get("text", "").startswith("🎙") for m, p in tg
               if m == "sendMessage"), "transcription is echoed"


# ── silent user continues normally ──────────────────────────────────────────


async def test_silent_user_continues_normally(tg, buffered):
    """After a restart, _feedback_pending is cleared (ephemeral) — the person
    who didn't answer just continues talking, and their message goes to the
    brain normally. feedback_asked persists, so the question is never repeated."""
    bot._contacts["1"] = {"feedback_asked": True, "turn_count": 3}
    bot._save_contacts()
    # _feedback_pending is empty — simulating a restart.

    await bot._handle(FakeClient(),
                      {"chat": {"id": 1}, "message_id": 1, "text": "продолжим"})

    assert len(buffered) == 1, "message goes to the brain, not captured as feedback"
    assert "продолжим" in buffered[0]


async def test_commands_still_work_while_feedback_pending(tg, monkeypatch):
    """Commands are not intercepted as feedback — the person can still /memory,
    /export etc. while the feedback window is open."""
    bot._contacts["1"] = {"turn_count": 3, "feedback_asked": True}
    bot._feedback_pending.add(1)

    await bot._handle(FakeClient(),
                      {"chat": {"id": 1}, "message_id": 1, "text": "/memory"})

    assert 1 in bot._feedback_pending, "command must not consume the feedback slot"
    assert any(m == "sendMessage" for m, _ in tg), "command was handled"


# ── persistence ──────────────────────────────────────────────────────────────


async def test_marker_survives_restart(tg, monkeypatch):
    await _run_turns(monkeypatch, 1, 3)
    assert bot._contacts["1"]["feedback_asked"] is True

    # Simulate a restart: RAM wiped, reload from disk.
    bot._contacts.clear()
    bot._feedback_pending.clear()
    bot._load_contacts()

    assert bot._contacts["1"]["feedback_asked"] is True
    assert bot._contacts["1"]["turn_count"] == 3


async def test_question_not_repeated_after_restart(tg, monkeypatch):
    await _run_turns(monkeypatch, 1, 3)
    assert _count_questions(tg) == 1

    # Restart: reload from disk, clear ephemeral state.
    bot._contacts.clear()
    bot._feedback_pending.clear()
    bot._chat.clear()
    bot._load_contacts()

    # Three more turns — question must not fire again.
    monkeypatch.setattr(bot, "DEBOUNCE_SEC", 0)
    monkeypatch.setattr(bot, "_ask_brain", _capture_turn([]))
    monkeypatch.setattr(bot, "_send_text", _noop)
    for i in range(3):
        st = bot._state(1)
        st["parts"] = [f"новый ход {i}"]
        await bot._worker(FakeClient(), 1)

    assert _count_questions(tg) == 1, "question must not be repeated after restart"


# ── /delete_my_data cleans up ────────────────────────────────────────────────


async def test_delete_clears_feedback_marker_and_answer(tg, monkeypatch):
    bot._contacts["42"] = {
        "feedback_asked": True, "turn_count": 3,
        "feedback": "супер коуч",
    }
    bot._save_contacts()
    bot._feedback_pending.add(42)

    async def fake_delete(tenant_id):
        return True
    monkeypatch.setattr(bot.memory, "delete_tenant", fake_delete)

    await bot._command(FakeClient(), 42, "/delete_confirm")

    assert "42" not in bot._contacts, "contact entry removed"
    assert 42 not in bot._feedback_pending, "pending flag cleared"

    # Reload from disk — gone there too.
    bot._contacts.clear()
    bot._load_contacts()
    assert "42" not in bot._contacts, "feedback data gone from disk"

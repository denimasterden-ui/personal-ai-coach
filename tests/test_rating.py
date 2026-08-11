"""Rating buttons — aicoach-dev#18.

Fake-Telegram harness, same pattern as test_bot_pdf.py.
All seven acceptance criteria are covered here.
"""
import asyncio
import time

import pytest

import bot
import supervision


# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(supervision, "DB_FILE", tmp_path / "sup.db")
    supervision.init()
    return supervision


@pytest.fixture(autouse=True)
def isolated_tenants(tmp_path, monkeypatch):
    import memory
    monkeypatch.setattr(memory.config, "TENANTS_DIR", tmp_path / "tenants")
    (tmp_path / "tenants").mkdir()


@pytest.fixture
def tg(monkeypatch):
    """Capture outbound Telegram calls; neuter access gating and analytics."""
    sent = []

    async def fake_tg(client, method, **params):
        sent.append((method, params))
        return {"ok": True}

    monkeypatch.setattr(bot, "_tg", fake_tg)
    monkeypatch.setattr(bot, "PUBLIC_MODE", True)
    monkeypatch.setattr(bot, "_rate_limited", lambda _: False)
    monkeypatch.setattr(bot.analytics, "has_seen", lambda _: True)
    monkeypatch.setattr(bot.analytics, "log", lambda *a, **k: None)
    return sent


class FakeClient:
    pass


def _text_msg(text, chat_id=7):
    return {"chat": {"id": chat_id}, "message_id": 1, "text": text}


def _cq(turn_id, value, chat_id=7, message_id=42):
    """Build a minimal callback_query dict."""
    return {
        "id": "cq1",
        "from": {"id": chat_id},
        "message": {"chat": {"id": chat_id}, "message_id": message_id},
        "data": f"rate:{turn_id}:{value}",
    }


def _keyboard_buttons(sent):
    """All button texts from sendMessage calls that carry an inline keyboard."""
    buttons = []
    for method, params in sent:
        if method == "sendMessage":
            rm = params.get("reply_markup")
            if rm:
                for row in rm.get("inline_keyboard", []):
                    buttons.extend(b["text"] for b in row)
    return buttons


async def _drain(chat_id, timeout=5.0):
    async def _wait():
        while True:
            st = bot._chat.get(chat_id)
            if st and not st["preparing"] and not st["parts"]:
                w = st["worker"]
                if w is None or w.done():
                    return
            await asyncio.sleep(0.05)
    await asyncio.wait_for(_wait(), timeout=timeout)


def _brain(text):
    async def _f(*a, **k):
        return text
    return _f


# ── AC1: buttons under разборы, not under service messages ───────────────────

async def test_coach_answer_has_rating_buttons(tg, monkeypatch, db):
    """Three rating buttons appear under each coaching разбор."""
    monkeypatch.setattr(bot, "DEBOUNCE_SEC", 0)
    monkeypatch.setattr(bot, "_ask_brain", _brain("разбор коуча"))

    await bot._handle(FakeClient(), _text_msg("расскажи что-то"))
    await _drain(7)

    labels = _keyboard_buttons(tg)
    assert "В точку" in labels
    assert "Частично" in labels
    assert "Мимо" in labels


async def test_service_messages_have_no_buttons(tg, monkeypatch):
    """Rate-limit reply must not carry rating buttons."""
    monkeypatch.setattr(bot, "_rate_limited", lambda _: True)
    await bot._handle(FakeClient(), _text_msg("что-то"))
    assert not _keyboard_buttons(tg), "service message must not get rating buttons"


async def test_brain_error_message_has_no_buttons(tg, monkeypatch, db):
    """Error fallback ('я на пару мгновений обновился') must not carry buttons."""
    monkeypatch.setattr(bot, "DEBOUNCE_SEC", 0)

    async def boom(*a, **k):
        raise RuntimeError("brain down")
    monkeypatch.setattr(bot, "_ask_brain", boom)

    await bot._handle(FakeClient(), _text_msg("привет"))
    await _drain(7)

    assert not _keyboard_buttons(tg)


# ── AC2: press saves rating tied to the specific turn ────────────────────────

async def test_button_press_saves_rating(tg, db):
    """Pressing a button writes the rating into supervision.db."""
    turn_id = "a" * 32
    await bot._handle_callback(FakeClient(), _cq(turn_id, "hit"))
    assert db.get_rating(turn_id) == "hit"


async def test_all_three_values_are_accepted(tg, db):
    ids = {"hit": "0" * 32, "partial": "1" * 32, "miss": "2" * 32}
    for value, turn_id in ids.items():
        await bot._handle_callback(FakeClient(), _cq(turn_id, value))
        assert db.get_rating(turn_id) == value


# ── AC3: re-press replaces rating, no duplicate row ──────────────────────────

async def test_re_press_replaces_rating(tg, db):
    """Second press with a different value replaces the first."""
    turn_id = "b" * 32
    await bot._handle_callback(FakeClient(), _cq(turn_id, "hit"))
    await bot._handle_callback(FakeClient(), _cq(turn_id, "miss"))
    assert db.get_rating(turn_id) == "miss"


async def test_re_press_does_not_duplicate(tg, db):
    """No second row is added on re-press."""
    turn_id = "c" * 32
    await bot._handle_callback(FakeClient(), _cq(turn_id, "hit"))
    await bot._handle_callback(FakeClient(), _cq(turn_id, "partial"))
    with db._conn() as c:
        count = c.execute("SELECT COUNT(*) FROM ratings WHERE turn_id = ?",
                          (turn_id,)).fetchone()[0]
    assert count == 1


# ── AC4: confirmation shown, no new message in the feed ──────────────────────

async def test_confirmation_shown_in_button(tg, db):
    """answerCallbackQuery carries a non-empty confirmation text."""
    turn_id = "d" * 32
    await bot._handle_callback(FakeClient(), _cq(turn_id, "partial"))
    ack = next((p for m, p in tg if m == "answerCallbackQuery"), None)
    assert ack is not None and ack.get("text"), "confirmation text must be non-empty"


async def test_no_new_chat_message_on_press(tg, db):
    """Pressing a button must not produce a sendMessage."""
    turn_id = "e" * 32
    await bot._handle_callback(FakeClient(), _cq(turn_id, "miss"))
    assert not any(m == "sendMessage" for m, _ in tg)


async def test_keyboard_updated_to_show_selection(tg, db):
    """editMessageReplyMarkup is called to reflect the chosen rating."""
    turn_id = "f" * 32
    await bot._handle_callback(FakeClient(), _cq(turn_id, "hit"))
    assert any(m == "editMessageReplyMarkup" for m, _ in tg)
    edit_params = next(p for m, p in tg if m == "editMessageReplyMarkup")
    buttons = [b["text"] for b in edit_params["reply_markup"]["inline_keyboard"][0]]
    assert any("✓" in b for b in buttons), "selected button must be marked"


# ── AC5: invalid payload — bot survives, nothing saved ───────────────────────

async def test_malformed_data_saves_nothing(tg, db):
    """Garbage payload: bot answers the callback but saves nothing."""
    cq = {"id": "bad1", "from": {"id": 7},
          "message": {"chat": {"id": 7}, "message_id": 1},
          "data": "not-valid-at-all"}
    await bot._handle_callback(FakeClient(), cq)
    assert any(m == "answerCallbackQuery" for m, _ in tg), "callback must still be acknowledged"
    with db._conn() as c:
        count = c.execute("SELECT COUNT(*) FROM ratings").fetchone()[0]
    assert count == 0


async def test_invalid_rating_value_saves_nothing(tg, db):
    """Valid turn_id format but unknown rating value: nothing saved."""
    turn_id = "3" * 32
    cq = _cq(turn_id, "supergood")
    await bot._handle_callback(FakeClient(), cq)
    assert db.get_rating(turn_id) is None


async def test_non_hex_turn_id_saves_nothing(tg, db):
    """Non-hexadecimal chars in turn_id: rejected."""
    cq = {"id": "bad2", "from": {"id": 7},
          "message": {"chat": {"id": 7}, "message_id": 1},
          "data": "rate:" + "x" * 32 + ":hit"}
    await bot._handle_callback(FakeClient(), cq)
    with db._conn() as c:
        count = c.execute("SELECT COUNT(*) FROM ratings").fetchone()[0]
    assert count == 0


async def test_short_turn_id_saves_nothing(tg, db):
    """turn_id shorter than 32 hex chars: rejected."""
    cq = {"id": "bad3", "from": {"id": 7},
          "message": {"chat": {"id": 7}, "message_id": 1},
          "data": "rate:abc:hit"}
    await bot._handle_callback(FakeClient(), cq)
    with db._conn() as c:
        count = c.execute("SELECT COUNT(*) FROM ratings").fetchone()[0]
    assert count == 0


async def test_missing_message_field_saves_nothing(tg, db):
    """callback_query without a .message (inline-mode edge case): no crash, no save."""
    cq = {"id": "bad4", "from": {"id": 7}, "data": "rate:" + "a" * 32 + ":hit"}
    await bot._handle_callback(FakeClient(), cq)
    with db._conn() as c:
        count = c.execute("SELECT COUNT(*) FROM ratings").fetchone()[0]
    assert count == 0


# ── AC6: rating readable together with turn trace ────────────────────────────

async def test_rating_joins_case_trace_via_turn_id(db):
    """pending() returns the rating alongside the case so critic sees both."""
    turn_id = "h" * 32
    now = time.time()
    # Insert a case with a known turn_id
    with db._conn() as c:
        c.execute(
            "INSERT INTO cases (ts, day, tenant, reasons, user_len, answer_len, "
            "tools, payload, turn_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (now, int(now // 86400), "alice", "onboarding", 100, 200, "",
             '{"user": "vopros", "answer": "otvet"}'.encode(), turn_id),
        )
    db.save_rating("alice", turn_id, "miss")

    cases = db.pending(tenant="alice")
    assert len(cases) == 1
    assert cases[0]["turn_id"] == turn_id
    assert cases[0]["rating"] == "miss"


async def test_unrated_case_has_none_rating(db):
    """Cases without a rating entry show rating=None (no row in ratings)."""
    now = time.time()
    with db._conn() as c:
        c.execute(
            "INSERT INTO cases (ts, day, tenant, reasons, user_len, answer_len, "
            "tools, payload, turn_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (now, int(now // 86400), "bob", "onboarding", 50, 100, "",
             b'{"user": "q", "answer": "a"}', "i" * 32),
        )
    cases = db.pending(tenant="bob")
    assert cases[0]["rating"] is None


async def test_miss_distinguishable_by_recall_count(db):
    """'miss' rating with empty recall vs. non-empty recall: both stored as 'miss',
    distinguishable via the case's tools column (AC wording)."""
    now = time.time()
    for i, tools in enumerate(("", "recall")):
        turn_id = chr(ord("j") + i) * 32
        with db._conn() as c:
            c.execute(
                "INSERT INTO cases (ts, day, tenant, reasons, user_len, answer_len, "
                "tools, payload, turn_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (now + i, int(now // 86400), "carol", "onboarding", 100, 200,
                 tools, b'{"user": "q", "answer": "a"}', turn_id),
            )
        db.save_rating("carol", turn_id, "miss")

    cases = {c["tools"]: c for c in db.pending(tenant="carol")}
    assert cases[""]["rating"] == "miss"
    assert cases["recall"]["rating"] == "miss"
    # They're the same rating value but distinguishable through the tools column
    assert cases[""]["tools"] != cases["recall"]["tools"]


# ── AC7: end-to-end through the harness ──────────────────────────────────────

async def test_e2e_buttons_rendered_press_saved_repress_updated(tg, monkeypatch, db):
    """Full scenario: buttons rendered → press saved → re-press replaced."""
    monkeypatch.setattr(bot, "DEBOUNCE_SEC", 0)
    captured_turn_id = []

    async def recording_brain(client, tenant, session, text, model=None,
                               turn_id=None, light_intro=False):
        captured_turn_id.append(turn_id)
        return "коучинговый разбор"

    monkeypatch.setattr(bot, "_ask_brain", recording_brain)

    # 1. Send a message and get a разбор with buttons
    await bot._handle(FakeClient(), _text_msg("что происходит?"))
    await _drain(7)

    labels = _keyboard_buttons(tg)
    assert "В точку" in labels and "Частично" in labels and "Мимо" in labels

    turn_id = captured_turn_id[0]
    assert turn_id and len(turn_id) == 32

    # 2. Press «В точку»
    tg.clear()
    await bot._handle_callback(FakeClient(), _cq(turn_id, "hit"))
    assert db.get_rating(turn_id) == "hit"
    assert not any(m == "sendMessage" for m, _ in tg), "press must not produce new message"

    # 3. Re-press «Мимо» — rating replaced
    tg.clear()
    await bot._handle_callback(FakeClient(), _cq(turn_id, "miss"))
    assert db.get_rating(turn_id) == "miss"
    with db._conn() as c:
        count = c.execute("SELECT COUNT(*) FROM ratings WHERE turn_id = ?",
                          (turn_id,)).fetchone()[0]
    assert count == 1, "re-press must not create a duplicate row"

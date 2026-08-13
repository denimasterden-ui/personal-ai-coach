"""Онбординговый тур (aicoach-dev#19).

Тур предлагается кнопкой после /start, идёт короткими шагами «дальше»/«хватит»,
прошедшему или прервавшему повторно не предлагается. Состояние хранится
в служебном хранилище продукта (_contacts), не в памяти тенанта (ADR 0004).
Все тесты через шов обвязки с фейковым Telegram.
"""
import asyncio

import pytest

import bot


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setattr(bot, "PRODUCT_STORE_FILE", tmp_path / "contacts.enc")
    bot._contacts.clear()


@pytest.fixture
def tg(monkeypatch):
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
    async def delete(self, url, **kw):
        return None


def _tour_offer_buttons(sent):
    """tour:* callback_data values found in sendMessage inline keyboards."""
    result = []
    for method, params in sent:
        if method == "sendMessage":
            rm = params.get("reply_markup") or {}
            for row in rm.get("inline_keyboard", []):
                for b in row:
                    cd = b.get("callback_data", "")
                    if cd.startswith("tour:"):
                        result.append(cd)
    return result


def _sent_texts(sent):
    return [p.get("text", "") for m, p in sent if m == "sendMessage"]


def _cq(action, chat_id=7):
    return {
        "id": "cq1",
        "from": {"id": chat_id},
        "message": {"chat": {"id": chat_id}, "message_id": 42},
        "data": f"tour:{action}",
    }


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


# ── AC1: тур предлагается кнопкой, не подменяет разбор ─────────────────────

async def test_start_offers_tour_button_to_new_user(tg):
    """/start carries a tour button when the user hasn't done the tour yet."""
    await bot._command(FakeClient(), 7, "/start")
    assert "tour:0" in _tour_offer_buttons(tg), "tour button must appear for a new user"


async def test_start_does_not_offer_button_after_tour_done(tg):
    """No tour button for a user who already completed or exited the tour."""
    bot._contacts["7"] = {"tour": "done"}
    await bot._command(FakeClient(), 7, "/start")
    assert not _tour_offer_buttons(tg), "tour button must not reappear after tour done"


async def test_разбор_carries_rating_buttons_not_tour_buttons(tg, monkeypatch):
    """The coaching разбор must carry rating buttons, never tour buttons."""
    monkeypatch.setattr(bot, "DEBOUNCE_SEC", 0)
    monkeypatch.setattr(bot, "_ask_brain",
                        lambda *a, **k: _always("разбор коуча"))
    monkeypatch.setattr(bot, "_send_text", _noop)

    await bot._handle(FakeClient(), {"chat": {"id": 7}, "message_id": 1, "text": "расскажи"})
    await _drain(7)

    assert not _tour_offer_buttons(tg), "tour buttons must not appear under a разбор"


# ── AC2: «дальше» двигает тур, «хватит» выходит ───────────────────────────

async def test_pressing_tour_button_shows_step0(tg):
    await bot._handle_callback(FakeClient(), _cq("0"))
    assert bot._TOUR_STEPS[0] in _sent_texts(tg), "step 0 text must be sent"


async def test_next_button_shows_step1(tg):
    await bot._handle_callback(FakeClient(), _cq("1"))
    assert bot._TOUR_STEPS[1] in _sent_texts(tg), "step 1 text must be sent"


async def test_step_message_has_nav_buttons(tg):
    """Each non-last step carries at least one navigation button."""
    await bot._handle_callback(FakeClient(), _cq("0"))
    step_buttons = [b for m, p in tg if m == "sendMessage"
                    for row in (p.get("reply_markup") or {}).get("inline_keyboard", [])
                    for b in row]
    assert step_buttons, "tour step must carry navigation buttons"


async def test_last_step_has_no_next_button(tg):
    """Last step must offer no «Дальше» — only the exit button."""
    last = len(bot._TOUR_STEPS) - 1
    await bot._handle_callback(FakeClient(), _cq(str(last)))
    cbs = [b.get("callback_data") for m, p in tg if m == "sendMessage"
           for row in (p.get("reply_markup") or {}).get("inline_keyboard", [])
           for b in row]
    assert f"tour:{last + 1}" not in cbs, "no forward button on the last step"
    assert "tour:exit" in cbs, "exit button must be on the last step"


async def test_exit_marks_tour_done(tg):
    await bot._handle_callback(FakeClient(), _cq("exit"))
    assert bot._contacts.get("7", {}).get("tour") == "done"


async def test_exit_acknowledged_with_text(tg):
    await bot._handle_callback(FakeClient(), _cq("exit"))
    ack = next((p for m, p in tg if m == "answerCallbackQuery"), None)
    assert ack and ack.get("text"), "exit must be acknowledged with a text"


# ── AC3: после выхода обычный разговор работает ────────────────────────────

async def test_conversation_works_after_tour_exit(tg, monkeypatch):
    """After tour exit a regular message goes to the brain normally."""
    bot._contacts["7"] = {"tour": "done"}

    monkeypatch.setattr(bot, "DEBOUNCE_SEC", 0)
    brain_calls = []

    async def recording_brain(client, tenant, session, text, model=None,
                               turn_id=None, light_intro=False):
        brain_calls.append(text)
        return "разбор"

    monkeypatch.setattr(bot, "_ask_brain", recording_brain)
    monkeypatch.setattr(bot, "_send_text", _noop)

    await bot._handle(FakeClient(), {"chat": {"id": 7}, "message_id": 1,
                                      "text": "расскажи про меня"})
    await _drain(7)
    assert brain_calls, "brain must be called after tour exit"
    assert "расскажи про меня" in brain_calls[0]


# ── AC4: повторно не предлагается ──────────────────────────────────────────

async def test_start_after_exit_has_no_tour_button(tg):
    """After exit, /start shows no tour button."""
    await bot._handle_callback(FakeClient(), _cq("exit"))
    tg.clear()
    await bot._command(FakeClient(), 7, "/start")
    assert not _tour_offer_buttons(tg)


async def test_stale_step_button_after_done_is_ignored(tg):
    """Old step button pressed after tour done: silently acknowledged, no new message."""
    bot._contacts["7"] = {"tour": "done"}
    await bot._handle_callback(FakeClient(), _cq("0"))
    assert not any(m == "sendMessage" for m, _ in tg), "no message for stale button"
    assert any(m == "answerCallbackQuery" for m, _ in tg), "callback acknowledged"


# ── AC5: состояние тура переживает перезапуск ──────────────────────────────

async def test_tour_done_survives_restart(tg):
    await bot._handle_callback(FakeClient(), _cq("exit"))
    assert bot._contacts.get("7", {}).get("tour") == "done"

    bot._contacts.clear()
    bot._load_contacts()
    assert bot._contacts.get("7", {}).get("tour") == "done", \
        "tour_done must persist across restart"


async def test_tour_not_offered_after_restart(tg):
    await bot._handle_callback(FakeClient(), _cq("exit"))
    bot._contacts.clear()
    bot._load_contacts()

    tg.clear()
    await bot._command(FakeClient(), 7, "/start")
    assert not _tour_offer_buttons(tg), "tour button must not reappear after restart"


# ── AC6: /delete_my_data сбрасывает состояние тура ────────────────────────

async def test_delete_resets_tour_state(tg, monkeypatch):
    bot._contacts["7"] = {"tour": "done"}
    bot._save_contacts()

    async def fake_delete(tenant_id):
        return True
    monkeypatch.setattr(bot.memory, "delete_tenant", fake_delete)

    await bot._command(FakeClient(), 7, "/delete_confirm")

    assert "7" not in bot._contacts, "contact entry removed"
    bot._contacts.clear()
    bot._load_contacts()
    assert "7" not in bot._contacts, "tour state gone from disk after delete"


# ── AC7: end-to-end — тур предложен, прошагал, прервался, не предложен ────

async def test_e2e_tour_offered_walked_exited_not_reoffered(tg):
    """Full scenario per AC7: offer → walk through steps → exit → no repeat."""
    # 1. /start → tour button appears
    await bot._command(FakeClient(), 7, "/start")
    assert "tour:0" in _tour_offer_buttons(tg), "tour button on first /start"

    # 2. Press tour button → step 0 shown
    tg.clear()
    await bot._handle_callback(FakeClient(), _cq("0"))
    assert bot._TOUR_STEPS[0] in _sent_texts(tg), "step 0 shown"

    # 3. Press Дальше → step 1 shown (if tour has more than one step)
    if len(bot._TOUR_STEPS) > 1:
        tg.clear()
        await bot._handle_callback(FakeClient(), _cq("1"))
        assert bot._TOUR_STEPS[1] in _sent_texts(tg), "step 1 shown"

    # 4. Press Хватит → tour exits and state persisted
    tg.clear()
    await bot._handle_callback(FakeClient(), _cq("exit"))
    assert bot._contacts.get("7", {}).get("tour") == "done", "tour marked done"

    # 5. /start again → no tour button
    tg.clear()
    await bot._command(FakeClient(), 7, "/start")
    assert not _tour_offer_buttons(tg), "tour not reoffered after exit"


# ── использованная кнопка гаснет ──────────────────────────────────────────────

def _cleared(sent):
    """message_id'ы, с которых сняли клавиатуру."""
    return [p["message_id"] for m, p in sent
            if m == "editMessageReplyMarkup" and not p["reply_markup"]["inline_keyboard"]]


async def test_used_tour_button_stops_being_tappable(tg):
    """Каждый шаг — отдельное сообщение, и без гашения чат остаётся усыпан
    живыми «Дальше →» от уже пройденных шагов, а оффер «Как это устроено?»
    жмётся и после конца тура."""
    await bot._handle_callback(FakeClient(), _cq("0"))
    assert 42 in _cleared(tg), "оффер должен погаснуть, когда тур начали"

    tg.clear()
    await bot._handle_callback(FakeClient(), _cq("1"))
    assert 42 in _cleared(tg), "пройденный шаг не должен остаться кликабельным"


async def test_last_step_button_goes_out_on_exit(tg):
    """«Понятно» — тоже использованная кнопка."""
    await bot._handle_callback(FakeClient(), _cq("exit"))
    assert 42 in _cleared(tg), "кнопка выхода должна погаснуть"


async def test_tour_survives_a_failed_clear(tg, monkeypatch):
    """Гашение — best-effort: в отличие от entry, здесь оно ничего не охраняет,
    поэтому сорвавшийся edit не должен съесть шаг тура."""
    async def flaky(client, method, **params):
        if method == "editMessageReplyMarkup":
            raise RuntimeError("message to edit not found")
        tg.append((method, params))
        return {"ok": True}
    monkeypatch.setattr(bot, "_tg", flaky)

    await bot._handle_callback(FakeClient(), _cq("0"))
    assert bot._TOUR_STEPS[0] in _sent_texts(tg), "шаг показан несмотря на сбой гашения"


# ── helpers ───────────────────────────────────────────────────────────────────

def _always(text):
    async def _f(*a, **k):
        return text
    return _f


async def _noop(*a, **k):
    return None

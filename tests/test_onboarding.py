"""Первое касание (R4 Г1/Г5): знакомство + ценность + 3 кнопки входа первым
сообщением, честные рамки вторым, событие onboarding_choice по нажатию.

Все тесты через шов обвязки с фейковым Telegram, как в test_tour.
"""
import pytest

import bot


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
    monkeypatch.setattr(bot.analytics, "has_seen", lambda _: True)
    monkeypatch.setattr(bot.analytics, "log", lambda *a, **k: None)
    return sent


class FakeClient:
    pass


def _messages(sent):
    return [p for m, p in sent if m == "sendMessage"]


def _entry_buttons(sent):
    out = []
    for m, p in sent:
        if m != "sendMessage":
            continue
        for row in (p.get("reply_markup") or {}).get("inline_keyboard", []):
            for b in row:
                cd = b.get("callback_data", "")
                if cd.startswith("entry:"):
                    out.append(cd)
    return out


def _callbacks(message):
    return [b.get("callback_data") for row in
            (message.get("reply_markup") or {}).get("inline_keyboard", []) for b in row]


def _entry_cq(choice, chat_id=7):
    return {
        "id": "cq1",
        "from": {"id": chat_id},
        "message": {"chat": {"id": chat_id}, "message_id": 42},
        "data": f"entry:{choice}",
    }


# ── такт ①: знакомство + ценность + кнопки, без отрицаний ──────────────────

async def test_start_sends_two_beats(tg):
    await bot._command(FakeClient(), 7, "/start")
    assert len(_messages(tg)) == 2, "первый контакт — два сообщения"


async def test_first_beat_introduces_and_leads_without_negation(tg):
    await bot._command(FakeClient(), 7, "/start")
    first = _messages(tg)[0]["text"]
    assert "коуч" in first.lower(), "первое сообщение должно представлять коуча"
    assert " не " not in first.lower(), "в первом такте не должно быть отрицаний"


async def test_first_beat_has_three_entry_buttons(tg):
    await bot._command(FakeClient(), 7, "/start")
    assert set(_entry_buttons(tg)) == {"entry:work", "entry:relations", "entry:decision"}


# ── такт ②: честные рамки + перенесённый тур ───────────────────────────────

async def test_second_beat_carries_the_disclaimer(tg):
    await bot._command(FakeClient(), 7, "/start")
    second = _messages(tg)[1]["text"]
    assert "не терапевт" in second and "/privacy" in second


async def test_tour_offer_moved_to_second_beat(tg):
    await bot._command(FakeClient(), 7, "/start")
    first, second = _messages(tg)[0], _messages(tg)[1]
    assert "tour:0" in _callbacks(second), "тур предлагается на втором сообщении"
    assert "tour:0" not in _callbacks(first), "первое сообщение несёт только вход, не тур"


# ── нажатие кнопки входа ───────────────────────────────────────────────────

async def test_entry_tap_logs_choice(tg, monkeypatch):
    logged = []
    monkeypatch.setattr(bot.analytics, "log",
                        lambda event, tenant, **kw: logged.append((event, kw.get("kind"))))
    await bot._handle_callback(FakeClient(), _entry_cq("work"))
    assert ("onboarding_choice", "work") in logged


async def test_entry_tap_sends_narrowing_prompt(tg):
    await bot._handle_callback(FakeClient(), _entry_cq("relations"))
    texts = [p["text"] for p in _messages(tg)]
    assert bot._ENTRY_PROMPTS["relations"] in texts


async def test_entry_tap_clears_keyboard_and_acknowledges(tg):
    await bot._handle_callback(FakeClient(), _entry_cq("decision"))
    methods = [m for m, _ in tg]
    assert "editMessageReplyMarkup" in methods, "клавиатура гасится после выбора"
    assert "answerCallbackQuery" in methods, "нажатие подтверждается"


async def test_entry_tap_does_not_log_if_keyboard_cannot_be_cleared(tg, monkeypatch):
    """Гашение клавиатуры — гейт лога: пока кнопки живы, повторный тап удвоил бы
    выбор. Если editMessageReplyMarkup упал, выбор не логируется и вопрос не
    шлётся — только ack, человек нажмёт снова (запишется один раз)."""
    logged = []
    monkeypatch.setattr(bot.analytics, "log",
                        lambda event, tenant, **kw: logged.append(event))

    async def fail_on_edit(client, method, **params):
        if method == "editMessageReplyMarkup":
            raise RuntimeError("message to edit not found")
        tg.append((method, params))
        return {"ok": True}
    monkeypatch.setattr(bot, "_tg", fail_on_edit)

    await bot._handle_callback(FakeClient(), _entry_cq("work"))

    assert "onboarding_choice" not in logged, "без гашения клавиатуры выбор не логируется"
    texts = [p.get("text", "") for m, p in tg if m == "sendMessage"]
    assert not any(t in bot._ENTRY_PROMPTS.values() for t in texts), "и вопрос не шлётся"
    assert "answerCallbackQuery" in [m for m, _ in tg], "но нажатие всё равно подтверждается"


async def test_invalid_entry_choice_is_ignored(tg, monkeypatch):
    logged = []
    monkeypatch.setattr(bot.analytics, "log",
                        lambda event, tenant, **kw: logged.append(event))
    await bot._handle_callback(FakeClient(), _entry_cq("nonsense"))
    assert "onboarding_choice" not in logged, "неизвестный вход не логируется как выбор"
    texts = [p.get("text", "") for m, p in tg if m == "sendMessage"]
    assert not any(t in bot._ENTRY_PROMPTS.values() for t in texts)


async def test_entry_tap_does_not_call_the_brain(tg, monkeypatch):
    """The narrowing question comes from a canned prompt, not a coaching turn —
    the person's next free-text message is what triggers _ask_brain."""
    async def fail_if_called(*a, **k):
        raise AssertionError("_ask_brain must not be called on an entry tap")
    monkeypatch.setattr(bot, "_ask_brain", fail_if_called)
    await bot._handle_callback(FakeClient(), _entry_cq("work"))
    texts = [p["text"] for p in _messages(tg)]
    assert bot._ENTRY_PROMPTS["work"] in texts

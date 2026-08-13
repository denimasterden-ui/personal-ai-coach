"""Сквозной journey: путь человека от /start до удаления памяти и возврата.

Отличие от остальных файлов: там проверяется, **что вернул хендлер**, здесь —
**что осталось на экране** после связки шагов. Оба дефекта ручного прогона
13.08.2026 жили именно между шагами, а не внутри них:
  · тур доигрывался корректно, но чат оставался усыпан живыми «Дальше →»;
  · /memory отвечал описью файлов, потому что тип loop переехал в loops/,
    а витрина осталась читать мёртвый open_loops.md.
Ни один из них не был виден с уровня «работает ли тур» — только с уровня пути.

Фейковый Telegram здесь с состоянием: он помнит отправленные сообщения и их
клавиатуры, поэтому можно спрашивать «какие кнопки в чате сейчас живые».
Модель не зовётся: мозг подменён, память пишется настоящими вызовами tools.
"""
import pytest

import bot
import memory
import supervision
import tools

CHAT = 7


# ── фейковый Telegram с состоянием чата ──────────────────────────────────────

class FakeChat:
    """Помнит сообщения и их клавиатуры — как настоящий клиент Telegram."""

    def __init__(self):
        self.messages = {}      # message_id -> {"text": str, "markup": dict|None}
        self.calls = []
        self._next_id = 100

    async def tg(self, client, method, **params):
        self.calls.append((method, params))
        if method == "sendMessage":
            self._next_id += 1
            self.messages[self._next_id] = {"text": params.get("text", ""),
                                            "markup": params.get("reply_markup")}
            return {"ok": True, "result": {"message_id": self._next_id}}
        if method == "editMessageReplyMarkup":
            mid = params["message_id"]
            if mid not in self.messages:
                raise RuntimeError("message to edit not found")
            self.messages[mid]["markup"] = params.get("reply_markup")
        return {"ok": True}

    # ── чем можно спрашивать состояние экрана ──
    def live_buttons(self):
        """callback_data всех кнопок, которые сейчас можно нажать в чате."""
        out = []
        for m in self.messages.values():
            for row in (m["markup"] or {}).get("inline_keyboard", []):
                out += [b["callback_data"] for b in row if "callback_data" in b]
        return out

    def texts(self):
        return [m["text"] for m in self.messages.values()]

    def last_with_buttons(self):
        for mid in sorted(self.messages, reverse=True):
            if (self.messages[mid]["markup"] or {}).get("inline_keyboard"):
                return mid, self.messages[mid]
        return None, None


class FakeClient:
    async def delete(self, url, **kw):
        return None


@pytest.fixture
def chat(tmp_path, monkeypatch):
    fake = FakeChat()
    monkeypatch.setattr(bot, "_tg", fake.tg)
    monkeypatch.setattr(bot, "PUBLIC_MODE", True)
    monkeypatch.setattr(bot, "DEBOUNCE_SEC", 0)
    monkeypatch.setattr(bot, "_rate_limited", lambda _: False)
    monkeypatch.setattr(bot.analytics, "log", lambda *a, **k: None)
    monkeypatch.setattr(bot.analytics, "has_seen", lambda _: True)
    monkeypatch.setattr(bot, "PRODUCT_STORE_FILE", tmp_path / "contacts.enc")
    monkeypatch.setattr(memory.config, "TENANTS_DIR", tmp_path / "tenants")
    (tmp_path / "tenants").mkdir()
    monkeypatch.setattr(supervision, "DB_FILE", tmp_path / "sup.db")
    supervision.init()
    return fake


# ── шаги пути ────────────────────────────────────────────────────────────────

async def _msg(text):
    await bot._handle(FakeClient(), {"chat": {"id": CHAT}, "message_id": 1, "text": text})
    await _drain()


async def _tap(data, message_id):
    await bot._handle_callback(FakeClient(), {
        "id": "cq", "from": {"id": CHAT},
        "message": {"chat": {"id": CHAT}, "message_id": message_id},
        "data": data})


async def _drain(timeout=5.0):
    import asyncio
    async def _wait():
        while True:
            st = bot._chat.get(CHAT)
            if st and not st["preparing"] and not st["parts"]:
                w = st["worker"]
                if w is None or w.done():
                    return
            elif not st:
                return
            await asyncio.sleep(0.02)
    await asyncio.wait_for(_wait(), timeout=timeout)


def _mid_of(chat, needle):
    """message_id сообщения, содержащего фрагмент текста."""
    return next(mid for mid, m in chat.messages.items() if needle in m["text"])


def _brain(answer="Разбор коуча.", writes=None):
    """Мозг-заглушка: отвечает и, если попросили, пишет память настоящим tools —
    как это делает живая модель через вызов инструмента."""
    async def _ask(client, tenant, session, text, **kw):
        if writes:
            await tools.dispatch("save_memory", dict(writes), tenant)
        return answer
    return _ask


# ── journey ──────────────────────────────────────────────────────────────────

async def test_first_session_journey(chat, monkeypatch):
    """Полный путь первой сессии. Каждый блок — шаг человека, а проверка после
    него — что он видит на экране."""

    # ① /start: знакомство с дверями + рамки с оффером тура
    await _msg("/start")
    assert len(chat.messages) == 2, "первое касание — два такта, не стена"
    assert set(chat.live_buttons()) == {
        "entry:work", "entry:relations", "entry:decision", "tour:0"}
    intro, frames = [chat.messages[m]["text"] for m in sorted(chat.messages)]
    assert " не " not in intro, "в приглашении не должно быть отрицаний — они во втором такте"
    assert "не терапевт" in frames

    # ② выбрал дверь: кнопки входа гаснут, приходит один сужающий вопрос
    monkeypatch.setattr(bot, "_ask_brain", _brain())
    before = len(chat.messages)
    await _tap("entry:relations", _mid_of(chat, "Привет"))
    assert chat.live_buttons() == ["tour:0"], "использованные двери не должны остаться живыми"
    assert len(chat.messages) == before + 1, "ровно один вопрос, без разбора"
    assert bot._ENTRY_PROMPTS["relations"] in chat.texts()

    # ③ тур целиком: после выхода в чате не остаётся ни одной живой кнопки тура
    await _tap("tour:0", _mid_of(chat, "честных рамок"))
    await _tap("tour:1", _mid_of(chat, bot._TOUR_STEPS[0][:20]))
    await _tap("tour:2", _mid_of(chat, bot._TOUR_STEPS[1][:20]))
    await _tap("tour:exit", _mid_of(chat, bot._TOUR_STEPS[2][:20]))
    assert not [b for b in chat.live_buttons() if b.startswith("tour:")], (
        "пройденный тур не должен оставлять кликабельные шаги — так и было в проде")
    assert bot._contacts[str(CHAT)]["tour"] == "done"

    # ④ три хода: под каждым разбором оценки, после третьего — вопрос о продукте
    monkeypatch.setattr(bot, "_ask_brain", _brain(
        writes={"type": "loop", "slug": "x5-канал", "status": "open",
                "content": "Выбран один канал захода в X5 — через бизнес-контакт.",
                "source": "сессия"}))
    for i in range(bot.FEEDBACK_AFTER_TURN):
        await _msg(f"ход номер {i}")
    rate_buttons = [b for b in chat.live_buttons() if b.startswith("rate:")]
    assert len(rate_buttons) == 3 * bot.FEEDBACK_AFTER_TURN, "оценки живут под каждым разбором"
    assert bot.FEEDBACK_QUESTION in chat.texts(), "после третьего хода спрашиваем о продукте"
    assert CHAT in bot._feedback_pending

    # ⑤ ответ на вопрос о продукте не уходит коучу
    monkeypatch.setattr(bot, "_ask_brain", _brain("ЭТОГО РАЗБОРА БЫТЬ НЕ ДОЛЖНО"))
    await _msg("кнопки удобные, тур длинноват")
    assert "ЭТОГО РАЗБОРА БЫТЬ НЕ ДОЛЖНО" not in chat.texts()
    assert bot.FEEDBACK_THANKS in chat.texts()

    # ⑥ оценка: передумал — галочка одна, последняя
    mid, msg = chat.last_with_buttons()
    turn_id = msg["markup"]["inline_keyboard"][0][0]["callback_data"].split(":")[1]
    await _tap(f"rate:{turn_id}:hit", mid)
    await _tap(f"rate:{turn_id}:miss", mid)
    marks = [b["text"] for row in chat.messages[mid]["markup"]["inline_keyboard"] for b in row]
    assert sum(m.startswith("✓") for m in marks) == 1, "галочка ровно одна"
    assert "✓ Мимо" in marks, "выигрывает последняя оценка"

    # ⑦ /memory показывает живую открытую тему, а не опись файлов
    await _msg("/memory")
    shown = " ".join(chat.texts())
    assert "через бизнес-контакт" in shown, (
        "открытая тема, записанная мозгом в loops/, должна быть видна человеку")
    assert "_status:" not in shown, "служебные поля наружу не текут"

    # ⑧ удаление: текста нет нигде, обезличенная оценка остаётся
    tenant = bot._tenant_id_for(CHAT)
    # мозг на своей стороне кладёт ход в базу качества — воспроизводим это,
    # чтобы удалению было что забывать
    supervision.capture(tenant, "то, что человек рассказал", "разбор коуча",
                        budget_exhausted=True, turn_id=turn_id)
    await _msg("/delete_my_data")
    await _msg("/delete_confirm")
    assert not (await memory.summarize(tenant)).startswith("Работает"), "память стёрта"
    with supervision._conn() as c:
        payloads = [r[0] for r in c.execute("SELECT payload FROM cases")]
        tenants = [r[0] for r in c.execute("SELECT tenant FROM cases")]
        ratings = [r[0] for r in c.execute("SELECT rating FROM ratings")]
    assert all(not p for p in payloads), "слова человека стёрты из базы качества"
    assert all(t == supervision.DELETED_TENANT for t in tenants), "кейсы обезличены"
    assert ratings == ["miss"], "урок остаётся: оценка пережила удаление"
    assert str(CHAT) not in bot._contacts, "продуктовое хранилище тоже забывает"
    kept = supervision.feedback()
    assert [f["text"] for f in kept] == ["кнопки удобные, тур длинноват"], (
        "отзыв о сервисе — показание о продукте, а не часть удаляемого разговора")
    assert kept[0]["chat_id"] == str(CHAT), "и остаётся возводимым к аккаунту"

    # ⑨ возврат: встречают как впервые
    chat.messages.clear()
    await _msg("/start")
    assert set(chat.live_buttons()) == {
        "entry:work", "entry:relations", "entry:decision", "tour:0"}, (
        "после удаления тур предлагается снова — состояния не осталось")

"""Первый контакт определяется записями памяти, а не каталогом tenant-а."""

import memory
import bot
import service
from types import SimpleNamespace


async def test_derived_memory_marks_a_tenant_as_returning(tmp_path, monkeypatch):
    monkeypatch.setattr(memory.config, "TENANTS_DIR", tmp_path / "tenants")
    tenant = "new-person"

    assert not await memory.has_derived_records(tenant)

    await memory.save_memory(tenant, "doc", "текст присланного отчёта",
                             slug="report", source="файл")
    assert not await memory.has_derived_records(tenant), "source document is not a derived record"

    await memory.save_memory(tenant, "pattern", "откладывает сложные разговоры",
                             slug="avoidance", source="вывод из сессии")
    assert await memory.has_derived_records(tenant)


async def test_bot_routes_short_and_substantive_first_turns_differently(monkeypatch):
    """The bot-to-brain seam carries a light introduction only for a greeting."""
    routes = []

    async def ask(_client, _tenant, _session, _text, model=None, turn_id=None, light_intro=False):
        routes.append(light_intro)
        return "ответ"

    monkeypatch.setattr(bot, "DEBOUNCE_SEC", 0)
    monkeypatch.setattr(bot.memory, "has_derived_records", _false)
    monkeypatch.setattr(bot, "_remember_first_contact", _noop)
    monkeypatch.setattr(bot, "_ask_brain", ask)
    monkeypatch.setattr(bot, "_send_text", _noop)
    monkeypatch.setattr(bot.analytics, "log", lambda *a, **kw: None)
    bot._chat.clear()

    for chat_id, text in ((1, "привет"), (2, "Я выгораю на работе и не понимаю, что менять.")):
        state = bot._state(chat_id)
        state["parts"] = [text]
        await bot._worker(None, chat_id)

    assert routes == [True, False]


async def test_compact_disclosure_is_not_mistaken_for_a_greeting(monkeypatch):
    monkeypatch.setattr(bot.memory, "has_derived_records", _false)

    assert not await bot._needs_light_intro("t1", "Меня уволили", attachment=False)


async def test_first_contact_marker_prevents_a_later_greeting_from_rerouting(tmp_path, monkeypatch):
    monkeypatch.setattr(memory.config, "TENANTS_DIR", tmp_path / "tenants")

    first, light_intro = await bot._first_contact_route("t1", "привет", attachment=False)
    assert (first, light_intro) == (True, True)
    await bot._remember_first_contact("t1")

    assert await bot._first_contact_route("t1", "привет", attachment=False) == (False, False)


async def _false(_tenant):
    return False


async def _noop(*args, **kwargs):
    return None


async def test_brain_uses_a_different_prompt_for_light_first_contact(monkeypatch):
    async def playbook(_title):
        return "# Кризис\nСначала отреагируй на острое состояние человека."

    async def catalog():
        return []

    monkeypatch.setattr(service.memory, "load_skill", playbook)
    monkeypatch.setattr(service.memory, "load_skills_catalog", catalog)

    first_prompt = await service._build_system_prompt(light_intro=True)
    returning_prompt = await service._build_system_prompt(light_intro=False)

    assert "# Лёгкое знакомство" in first_prompt
    assert "# Лёгкое знакомство" not in returning_prompt
    assert first_prompt.index("# Кризис") < first_prompt.index("# Лёгкое знакомство")


async def test_attachment_and_returning_memory_skip_light_intro(monkeypatch):
    async def returning(_tenant):
        return True

    assert not await bot._needs_light_intro("t1", "привет", attachment=True)

    monkeypatch.setattr(bot.memory, "has_derived_records", returning)
    assert not await bot._needs_light_intro("t1", "привет", attachment=False)


async def test_light_intro_is_removed_after_the_first_turn(monkeypatch):
    calls = []

    class Message:
        content = "ответ"
        tool_calls = None

        def model_dump(self, exclude_none=True):
            return {"role": "assistant", "content": self.content}

    async def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=Message())])

    async def playbook(_title):
        return "# Кризис"

    async def catalog():
        return []

    monkeypatch.setattr(service, "_client", SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    monkeypatch.setattr(service.memory, "load_skill", playbook)
    monkeypatch.setattr(service.memory, "load_skills_catalog", catalog)
    monkeypatch.setattr(service.supervision, "capture_async", lambda *args, **kwargs: None)
    monkeypatch.setattr(service.recollect, "after_turn_async", lambda *args, **kwargs: None)
    service._sessions.clear()

    async for _ in service._run_inner(service.SessionRequest(
        tenant_id="t1", session_id="same", message="привет", light_intro=True)):
        pass
    async for _ in service._run_inner(service.SessionRequest(
        tenant_id="t1", session_id="same", message="продолжим")):
        pass

    second_system = calls[1]["messages"][0]["content"][0]["text"]
    assert "# Лёгкое знакомство" not in second_system

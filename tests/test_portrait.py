"""Portrait generation through the brain's public endpoint seam."""

from types import SimpleNamespace

import pytest

import bot
import memory
import service


class FakeClient:
    def __init__(self, text):
        self.text = text
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=self.text),
        )])


@pytest.fixture
def tenant(tmp_path, monkeypatch):
    monkeypatch.setattr(memory.config, "TENANTS_DIR", tmp_path / "tenants")
    (tmp_path / "tenants").mkdir()
    return "person"


async def test_portrait_refuses_when_memory_has_too_few_derived_entries(tenant, monkeypatch):
    """Below the evidence threshold, no model call is spent and the reason is usable."""
    client = FakeClient("this must not be returned")
    monkeypatch.setattr(service, "_client", client)
    await memory.save_memory(tenant, "pattern", "Сначала наблюдает", slug="observe")
    await memory.save_memory(tenant, "pattern", "Любит ясность", slug="clarity")

    result = await service.portrait(service.PortraitRequest(tenant_id=tenant))

    assert result["ready"] is False
    assert "ещё рано" in result["reason"]
    assert "5" in result["reason"] and "3" in result["reason"]
    assert client.calls == []


async def test_portrait_refuses_when_threshold_met_only_by_brought_material(tenant, monkeypatch):
    """doc (uploaded file) and session (transcript) are brought material, not
    derived findings (CONTEXT.md) — five of them alone must not satisfy the
    threshold, even though save_memory happily accepts them."""
    client = FakeClient("this must not be returned")
    monkeypatch.setattr(service, "_client", client)
    for slug in ("a", "b", "c", "d", "e"):
        await memory.save_memory(tenant, "doc", f"файл {slug}", slug=slug)

    result = await service.portrait(service.PortraitRequest(tenant_id=tenant))

    assert result["ready"] is False
    assert client.calls == []


async def test_portrait_generates_from_full_dump_without_writing_memory(tenant, monkeypatch):
    """A qualifying portrait is one tool-free model call over the export dump."""
    client = FakeClient("Связный портрет человека")
    monkeypatch.setattr(service, "_client", client)
    for slug in ("observe", "clarity", "challenge"):
        await memory.save_memory(tenant, "pattern", f"Паттерн {slug}", slug=slug)
    await memory.save_memory(tenant, "coach", "Наблюдение коуча", slug="note")
    await memory.save_memory(tenant, "decision", "Решение по итогам", slug="commit")
    await memory.save_memory(tenant, "doc", "ПОЛНЫЙ ТЕКСТ ПРИСЛАННОГО ФАЙЛА", slug="report")
    before = await memory.export_tenant(tenant)

    result = await service.portrait(service.PortraitRequest(tenant_id=tenant))

    assert result == {"ready": True, "text": "Связный портрет человека"}
    assert len(client.calls) == 1
    assert "tools" not in client.calls[0]
    assert "ПОЛНЫЙ ТЕКСТ ПРИСЛАННОГО ФАЙЛА" in client.calls[0]["messages"][1]["content"]
    assert await memory.export_tenant(tenant) == before
    assert service._sessions == {}


async def test_portrait_is_regenerated_from_current_memory_each_time(tenant, monkeypatch):
    client = FakeClient("Первая версия")
    monkeypatch.setattr(service, "_client", client)
    for slug in ("one", "two", "three"):
        await memory.save_memory(tenant, "pattern", slug, slug=slug)
    await memory.save_memory(tenant, "coach", "наблюдение", slug="note")
    await memory.save_memory(tenant, "test", "результат теста", slug="assessment")
    await memory.save_memory(tenant, "doc", "первый источник", slug="source")

    first = await service.portrait(service.PortraitRequest(tenant_id=tenant))
    await memory.save_memory(tenant, "decision", "НОВОЕ АКТУАЛЬНОЕ РЕШЕНИЕ", slug="next")
    client.text = "Вторая версия"
    second = await service.portrait(service.PortraitRequest(tenant_id=tenant))

    assert first["text"] == "Первая версия"
    assert second["text"] == "Вторая версия"
    assert len(client.calls) == 2
    assert "НОВОЕ АКТУАЛЬНОЕ РЕШЕНИЕ" not in client.calls[0]["messages"][1]["content"]
    assert "НОВОЕ АКТУАЛЬНОЕ РЕШЕНИЕ" in client.calls[1]["messages"][1]["content"]


async def test_portrait_reports_failure_when_model_returns_no_text(tenant, monkeypatch):
    client = FakeClient("")
    monkeypatch.setattr(service, "_client", client)
    for slug in ("one", "two", "three"):
        await memory.save_memory(tenant, "pattern", slug, slug=slug)
    await memory.save_memory(tenant, "coach", "наблюдение", slug="note")
    await memory.save_memory(tenant, "test", "результат теста", slug="assessment")
    await memory.save_memory(tenant, "doc", "источник", slug="source")

    result = await service.portrait(service.PortraitRequest(tenant_id=tenant))

    assert result["ready"] is False
    assert result["reason"]


class FakeTelegramClient:
    def __init__(self):
        self.posts = []

    async def post(self, url, **kwargs):
        self.posts.append((url, kwargs))
        return SimpleNamespace(json=lambda: {"ok": True})


async def test_portrait_command_sends_text_and_markdown_file(monkeypatch):
    sent = []

    async def fake_tg(client, method, **params):
        sent.append((method, params))
        return {"ok": True}

    async def fake_portrait(client, tenant):
        return {"ready": True, "text": "Твой связный портрет"}

    monkeypatch.setattr(bot, "_tg", fake_tg)
    monkeypatch.setattr(bot, "_ask_portrait", fake_portrait)
    monkeypatch.setattr(bot.analytics, "has_seen", lambda _: True)
    monkeypatch.setattr(bot.analytics, "log", lambda *a, **k: None)
    client = FakeTelegramClient()

    await bot._command(client, 7, "/portrait")

    assert sent == [("sendMessage", {"chat_id": 7, "text": "Твой связный портрет"})]
    assert client.posts[0][1]["files"]["document"][0] == "aicoach-portrait.md"
    assert "Твой связный портрет" in client.posts[0][1]["files"]["document"][1].decode()


async def test_portrait_command_shows_too_early_reason(monkeypatch):
    sent = []

    async def fake_tg(client, method, **params):
        sent.append((method, params))
        return {"ok": True}

    async def fake_portrait(client, tenant):
        return {"ready": False, "reason": "Для портрета ещё рано."}

    monkeypatch.setattr(bot, "_tg", fake_tg)
    monkeypatch.setattr(bot, "_ask_portrait", fake_portrait)
    monkeypatch.setattr(bot.analytics, "has_seen", lambda _: True)
    monkeypatch.setattr(bot.analytics, "log", lambda *a, **k: None)

    await bot._command(FakeTelegramClient(), 7, "/portrait")

    assert sent == [("sendMessage", {"chat_id": 7, "text": "Для портрета ещё рано."})]

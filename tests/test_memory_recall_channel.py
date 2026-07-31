"""memory.py — отдельный always-on recall-канал для открытых петель.

До фикса recall смешивал в одном top-K по count две разные семантики:
обязательный контекст (self.md + все открытые петли) и релевантное запросу
(patterns, sessions...). После миграции петель стало бы 5+, и они вместе с
self.md вытеснили бы остальное. Лечение: открытые петли (_status: open_) выходят
в отдельный канал, ограниченный по объёму (_OPEN_LOOPS_CHAR_CAP), а не по count;
свежие важнее залежавшихся; при переполнении — заглушка, чтобы модель знала, что
канал обрезан, а не исчерпан.
"""
import re

import pytest

import memory
import tools


@pytest.fixture(autouse=True)
def isolated_tenants(tmp_path, monkeypatch):
    """Изолируем config.TENANTS_DIR в tmp_path — тесты, пишущие в живой tenants/,
    на этом проекте уже случались. Скопировано из tests/test_memory_loops.py."""
    monkeypatch.setattr(memory.config, "TENANTS_DIR", tmp_path / "tenants")
    (tmp_path / "tenants").mkdir()


TENANT = "channel-tenant"


async def _loop(slug, status, content, tenant=TENANT, source="сессия 2026-07"):
    await tools.dispatch("save_memory",
                         {"type": "loop", "slug": slug, "status": status,
                          "content": content, "source": source}, tenant)


def _age_loop(slug, stamp="2020-01-01T00:00:00+00:00", tenant=TENANT):
    """Состарить петлю: тестовый авто-_updated_ ставит всем петлям почти одинаковый
    штамп, поэтому без правки футера проверить «свежая остаётся, старая отсекается»
    нельзя. Прямая правка файла в изолированном tmp_path."""
    path = memory._tenant_dir(tenant) / "loops" / f"{slug}.md"
    text = memory._read_text(path)
    memory._write_text(path, re.sub(r"_updated: \S+?_", f"_updated: {stamp}_", text))


def _loops(hits):
    return [h for h in hits if h.get("path", "").startswith("loops/") and h["path"] != "loops/"]


async def test_all_open_loops_in_output_beside_query_relevant(tmp_path):
    """7 открытых петель + self.md + 2 pattern. Нерелевантный петлям запрос ->
    в выдаче все 7 петель (канал не зависит от query-score) плюс pattern.
    До фикса петель в выдаче было не больше limit (5)."""
    for i in range(7):
        await _loop(f"loop-{i}", "open", f"открытая тема номер {i}")
    await memory.save_memory(TENANT, "self", "профиль пользователя", source="онбординг")
    await tools.dispatch("save_memory",
                         {"type": "pattern", "slug": "sleep", "content": "паттерн про сон",
                          "source": "сессия"}, TENANT)
    await tools.dispatch("save_memory",
                         {"type": "pattern", "slug": "focus", "content": "ещё один сон-паттерн",
                          "source": "сессия"}, TENANT)
    hits = await memory.recall(TENANT, "сон")
    loops = _loops(hits)
    assert len(loops) == 7, "все открытые петли — в always-on канале, без потолка limit"
    assert any(h["path"].startswith("patterns/") for h in hits), "pattern не вытеснен петлями"


async def test_cap_truncates_with_stub(tmp_path):
    """Суммарный объём открытых петель > cap -> петельный content в выдаче уложен в
    cap, и есть заглушка о числе отсечённых."""
    big = "надо согласовать слайды и текст " * 60  # ~1.7k символов -> 5 петель > cap(4000)
    for i in range(5):
        await _loop(f"big-{i}", "open", big)
    hits = await memory.recall(TENANT, "несуществующееслово")
    loops = _loops(hits)
    stub = [h for h in hits if h.get("path") == "loops/"]
    assert stub, "при переполнении обязательна заглушка — молчаливое отсечение это исходный баг"
    assert "+ ещё" in stub[0]["content"]
    total = sum(len(h["content"]) for h in loops)
    assert total <= memory._OPEN_LOOPS_CHAR_CAP, "канал уложен в cap по объёму"
    assert len(loops) < 5, "часть петель отсечена по cap"


async def test_freshness_wins_on_truncation(tmp_path):
    """Две открытые петли, у одной _updated заметно старее. При превышении cap в
    выдаче остаётся свежая, отсекается старая."""
    big = "очень длинное содержание петли " * 80  # > cap/2 -> две вместе превышают cap
    await _loop("fresh", "open", big)
    await _loop("stale", "open", big)
    _age_loop("stale")  # 2020 -> старше
    hits = await memory.recall(TENANT, "несуществующееслово")
    loops = _loops(hits)
    assert [h["path"] for h in loops] == ["loops/fresh.md"], \
        "при нехватке места остаётся свежая петля, старая отсекается"


async def test_edited_loop_keeps_its_freshness(tmp_path):
    """edit_memory пишет футер в другой форме — `_updated: <iso> (правка: …)_`.
    Если разбор штампа её не понимает, поправленная петля читается как недатированная
    и отсекается первой, хотя она самая свежая."""
    big = "очень длинное содержание петли " * 80
    await _loop("edited", "open", big + " ЗАМЕНИТЬ")
    await _loop("other", "open", big)
    _age_loop("other")
    await memory.edit_memory(TENANT, "loop", "ЗАМЕНИТЬ", "исправлено",
                             slug="edited", source="слова человека")
    hits = await memory.recall(TENANT, "несуществующееслово")
    assert [h["path"] for h in _loops(hits)] == ["loops/edited.md"], \
        "поправленная петля — самая свежая, отсекаться должна не она"


async def test_done_dropped_not_in_always_on_channel(tmp_path):
    """done/dropped — обычный query-relevant контент, в always-on блок не попадают.
    По нерелевантному запросу готовой петли в выдаче нет."""
    await _loop("finished", "done", "завершённая ранее задача")
    hits = await memory.recall(TENANT, "несуществующееслово")
    assert not _loops(hits), "done-петля не в always-on канале; score 0 -> вне выдачи"
    assert not any("finished" in h.get("path", "") for h in hits)


async def test_query_relevant_not_displaced(tmp_path):
    """Pattern, релевантный запросу, находится в выдаче рядом с open-петлями —
    канал не сожрал весь top-K."""
    await _loop("a", "open", "тема петли")
    await _loop("b", "open", "тема петли")
    await tools.dispatch("save_memory",
                         {"type": "pattern", "slug": "sleep", "content": "паттерн про сон",
                          "source": "сессия"}, TENANT)
    hits = await memory.recall(TENANT, "сон")
    assert any(h["path"] == "patterns/sleep.md" for h in hits)


async def test_contract_every_entry_has_path_and_content(tmp_path):
    """Контракт recall не изменился: list[dict] с ключами path и content (str) —
    на него полагается tools.py и тест T1."""
    await _loop("a", "open", "тема петли")
    await tools.dispatch("save_memory",
                         {"type": "pattern", "slug": "sleep", "content": "паттерн про сон",
                          "source": "сессия"}, TENANT)
    hits = await memory.recall(TENANT, "сон")
    assert hits, "выдача не пуста"
    for h in hits:
        assert set(("path", "content")).issubset(h), "каждый элемент имеет path и content"
        assert isinstance(h["path"], str) and isinstance(h["content"], str)

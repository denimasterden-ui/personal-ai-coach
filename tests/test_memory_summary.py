"""'/memory' — витрина того, что коуч держит о человеке.

Регрессия, найденная в ручном прогоне: тип loop переехал в каталог loops/, а
витрина осталась читать единственный файл open_loops.md, который для записи уже
запрещён. Три живые открытые темы не показывались вообще. Паттерны при этом были
видны только числом («1 паттернов») — на вопрос «что ты про меня понял» витрина
отвечала описью файловой системы.
"""
import pytest

import memory
import tools


@pytest.fixture(autouse=True)
def isolated_tenants(tmp_path, monkeypatch):
    monkeypatch.setattr(memory.config, "TENANTS_DIR", tmp_path / "tenants")
    (tmp_path / "tenants").mkdir()


TENANT = "summary-tenant"


async def _loop(slug, content, status="open"):
    return await tools.dispatch(
        "save_memory",
        {"type": "loop", "slug": slug, "status": status, "content": content,
         "source": "сессия"}, TENANT)


async def _pattern(slug, content):
    return await tools.dispatch(
        "save_memory",
        {"type": "pattern", "slug": slug, "content": content, "source": "сессия"},
        TENANT)


async def test_open_loops_are_shown_not_hidden():
    """Главное: открытая тема из loops/ попадает в /memory текстом."""
    await _loop("x5-one-channel", "Выбран один канал захода в X5 — через бизнес-контакт.")

    out = await memory.summarize(TENANT)

    assert "через бизнес-контакт" in out, "открытая тема должна быть видна в /memory"
    assert "Открытые темы" in out


async def test_closed_loop_leaves_the_shelf():
    """Закрытая петля — уже не «открытая тема»: она не должна висеть в витрине
    как незакрытый долг."""
    await _loop("done-thing", "Созвон с операционщиками состоялся", status="done")

    out = await memory.summarize(TENANT)

    assert "Созвон с операционщиками" not in out
    assert "Открытые темы" not in out


async def test_legacy_open_loops_file_still_surfaces():
    """open_loops.md запрещён для записи, но у давних тенантов лежит на диске —
    переезд витрины на loops/ не должен спрятать то, что уже записано."""
    d = memory._tenant_dir(TENANT)
    d.mkdir(parents=True, exist_ok=True)
    (d / "open_loops.md").write_text("1. Старая тема из легаси-файла\n", encoding="utf-8")

    out = await memory.summarize(TENANT)

    assert "Старая тема из легаси-файла" in out


async def test_patterns_are_shown_as_text_not_as_a_count():
    """«1 паттернов» — факт о нашей файловой системе, а не ответ человеку."""
    await _pattern("premature-pivot", "Бросаешь линию, как только собеседник замолчал.")

    out = await memory.summarize(TENANT)

    assert "как только собеседник замолчал" in out
    assert "1 паттернов" not in out, "паттерн показан текстом, а не в описи"


async def test_long_profile_is_capped_and_says_so():
    """У активного человека витрина не должна превращаться в простыню: свежие
    целиком, остальные — числом, с указанием, где взять всё."""
    for i in range(_over := memory._SUMMARY_ENTRIES + 2):
        await _pattern(f"pattern-{i}", f"Наблюдение номер {i}")

    out = await memory.summarize(TENANT)

    shown = sum(f"Наблюдение номер {i}" in out for i in range(_over))
    assert shown == memory._SUMMARY_ENTRIES, "показываем ровно потолок"
    assert "и ещё 2" in out and "/export" in out


async def test_machine_fields_do_not_leak_into_the_view():
    """_status:/_updated: — служебные поля кода, человеку в /memory они мусор."""
    await _loop("visible", "Тема, которую видно")

    out = await memory.summarize(TENANT)

    assert "_status:" not in out and "_updated:" not in out


async def test_bulk_types_stay_counted():
    """Сессии и решения остаются числом — там число и есть честный ответ."""
    await tools.dispatch(
        "save_memory",
        {"type": "session", "slug": "s1", "content": "разбор от 13.08", "source": "сессия"},
        TENANT)

    out = await memory.summarize(TENANT)

    assert "1 сессий" in out
    assert "разбор от 13.08" not in out, "объём сессий не вываливаем в витрину"


async def test_empty_profile_still_invites():
    out = await memory.summarize(TENANT)
    assert "Пока я почти ничего о тебе не записал" in out

"""tools.dispatch — open_loops закрыт для записи/правок коучем (T5b).

Живой прогон T1+T4+T3 показал: даже с плейбук-правилом «новая петля → type=loop»
коуч под нагрузкой падает обратно в open_loops и плодит там нумерованные дубли —
ровно рецидив кейса PROD-1. Плейбук это ров, а не движок (спайк Hermes): единственное,
что держит — структурный гейт в dispatch, не проза. open_loops остаётся в enum, потому
что recall читает легаси open_loops.md; закрыт только путь записи коучем.
"""
import pytest

import memory
import tools


@pytest.fixture(autouse=True)
def isolated_tenants(tmp_path, monkeypatch):
    monkeypatch.setattr(memory.config, "TENANTS_DIR", tmp_path / "tenants")
    (tmp_path / "tenants").mkdir()


TENANT = "deprec-tenant"


async def test_save_open_loops_refused_with_loop_hint(tmp_path):
    """save_memory(type=open_loops) — отказ с подсказкой на loop. Файл не создаётся.
    Без гейта коуч снова свалил бы сюда новую задачу (рецидив PROD-1)."""
    res = await tools.dispatch(
        "save_memory",
        {"type": "open_loops", "content": "новая задача", "source": "сессия"}, TENANT)
    assert "error" in res and "устарел" in res["error"]
    assert not (memory._tenant_dir(TENANT) / "open_loops.md").exists(), \
        "при отказе файл не пишется"


async def test_edit_open_loops_refused(tmp_path):
    """edit_memory(type=open_loops) — тоже отказ, иначе правка стала бы лазейкой
    вместо save для обхода гейта."""
    res = await tools.dispatch(
        "edit_memory",
        {"type": "open_loops", "old_string": "старое", "new_string": "новое",
         "source": "правка"}, TENANT)
    assert "error" in res and "устарел" in res["error"]


async def test_recall_still_reads_legacy_open_loops(tmp_path):
    """Гейт закрыл запись, но не чтение: легаси open_loops.md, записанный напрямую
    (миграция/бот), по-прежнему находится в recall по слову из тела."""
    await memory.save_memory(TENANT, "open_loops",
                             "легаси тема уникальноеслово", source="старая сессия")
    hits = await memory.recall(TENANT, "уникальноеслово")
    assert any("уникальноеслово" in h["content"] for h in hits), \
        "чтение легаси open_loops не должно пострадать от гейта записи"

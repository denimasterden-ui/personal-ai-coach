"""memory.py — тип loop: петля со стабильным slug'ом и машиночитаемым статусом.

Корень баги был в том, что порядковый номер в нумерованном списке работал как
идентификатор — артефакт рендера стал ключом. У петли появляется slug (файл
loops/<slug>.md), статус становится полем _status:, которое пишет код, номера
уходят из данных. Плюс три гейта: статус обязателен для loop, модель не пишет
служебные строки (_source:/_updated:) в тело, а supersedes на single-file типе
честно отказывает вместо тихого no-op.
"""
import pytest

import memory
import tools


@pytest.fixture(autouse=True)
def isolated_tenants(tmp_path, monkeypatch):
    """Изолируем config.TENANTS_DIR в tmp_path — тесты, пишущие в живой tenants/,
    на этом проекте уже случались. Скопировано из tests/test_bot_pdf.py."""
    monkeypatch.setattr(memory.config, "TENANTS_DIR", tmp_path / "tenants")
    (tmp_path / "tenants").mkdir()


TENANT = "loops-tenant"


async def test_loop_save_creates_file_with_machine_readable_status(tmp_path):
    """Критерий: петля живёт в loops/<slug>.md, статус пишет код строкой _status:."""
    res = await tools.dispatch(
        "save_memory",
        {"type": "loop", "slug": "x5-presentation", "status": "open",
         "content": "Подготовить презентацию для X5", "source": "сессия 2026-07"},
        TENANT)
    assert "error" not in res
    loop_file = memory._tenant_dir(TENANT) / "loops" / "x5-presentation.md"
    assert loop_file.exists(), "петля должна лежать в loops/<slug>.md"
    assert "_status: open_" in memory._read_text(loop_file)


async def test_loop_resave_replaces_status_does_not_append_second(tmp_path):
    """Повторная запись с тем же slug заменяет файл: ровно один _status:, и он новый.
    Двух записей о петле (как у накопленных перекрывающихся списков) быть не должно."""
    await tools.dispatch(
        "save_memory",
        {"type": "loop", "slug": "x5-presentation", "status": "open",
         "content": "Подготовить презентацию", "source": "сессия 2026-07"}, TENANT)
    await tools.dispatch(
        "save_memory",
        {"type": "loop", "slug": "x5-presentation", "status": "done",
         "content": "Подготовить презентацию", "source": "сессия 2026-08"}, TENANT)
    text = memory._read_text(memory._tenant_dir(TENANT) / "loops" / "x5-presentation.md")
    assert text.count("_status:") == 1, "повторная запись заменяет, а не дописывает вторую"
    assert "_status: done_" in text


async def test_loop_without_slug_is_refused(tmp_path):
    """Новый per-entry тип должен попасть под существующий slug-гейт автоматически —
    без slug вторая петля затёрла бы первую."""
    res = await tools.dispatch(
        "save_memory",
        {"type": "loop", "status": "open", "content": "петля", "source": "сессия"}, "t1")
    assert "error" in res and "slug" in res["error"]
    assert not list(tmp_path.rglob("*.md"))


async def test_loop_without_status_is_refused(tmp_path):
    """Статус — обязательное поле петли; без него петля неотличима от записи без типа."""
    res = await tools.dispatch(
        "save_memory",
        {"type": "loop", "slug": "x5", "content": "петля", "source": "сессия"}, "t1")
    assert "error" in res and "status" in res["error"]
    assert not list(tmp_path.rglob("*.md"))


async def test_loop_invalid_status_lists_allowed_values(tmp_path):
    """Отказ должен перечислять допустимые значения — иначе модель не знает, что ввести."""
    res = await tools.dispatch(
        "save_memory",
        {"type": "loop", "slug": "x5", "status": "выполнено",
         "content": "петля", "source": "сессия"}, "t1")
    assert "error" in res
    assert all(v in res["error"] for v in ("open", "done", "dropped")), \
        "отказ должен перечислять допустимые значения"
    assert not list(tmp_path.rglob("*.md"))


async def test_save_rejects_a_forged_updated_line(tmp_path):
    """Модель вписывала свой _updated: с выдуманной датой выше настоящего футера —
    ложный признак актуальности. Гейт в save_memory отклоняет, файл не создаётся."""
    res = await memory.save_memory(
        "t1", "self", "утверждение\n_updated: 2026-08-01T00:00:00+00:00_", source="сессия")
    assert "error" in res and "_updated:" in res["error"]
    assert not list(tmp_path.rglob("*.md")), "файл не должен быть создан"


async def test_save_rejects_a_forged_status_line(tmp_path):
    """_status: тоже пишет код. Подделка в content ставит вторую строку статуса
    рядом с настоящей — петля снова становится двусмысленной."""
    res = await tools.dispatch(
        "save_memory",
        {"type": "loop", "slug": "x5", "status": "open", "source": "сессия",
         "content": "петля\n_status: done_"}, "t1")
    assert "error" in res and "_status:" in res["error"]
    assert not list(tmp_path.rglob("*.md")), "файл не должен быть создан"


async def test_edit_rejects_a_forged_source_line_in_new_string(tmp_path):
    """Та же защита в edit_memory: модель не подделывает служебные строки в new_string."""
    await memory.save_memory("t1", "self", "тема для правки", source="онбординг")
    res = await memory.edit_memory(
        "t1", "self", "тема для правки",
        "тема для правки\n_source: тест Gallup 2021_", source="правка")
    assert "error" in res and "_source:" in res["error"]
    text = memory._read_text(memory._tenant_dir("t1") / "self.md")
    assert "тема для правки" in text, "отказ не должен портить существующую запись"
    assert "Gallup 2021" not in text, "поддельная строка не должна попасть в файл"


async def test_supersedes_on_single_file_type_is_an_error_not_silent_success(tmp_path):
    """Для single-file типа _resolve возвращает тот же путь, и supersedes молча
    ничего не делал, отчитываясь успехом. Теперь честный отказ с подсказкой как надо."""
    res = await memory.save_memory("t1", "open_loops", "тема",
                                   supersedes=["что-то"], source="сессия")
    assert "error" in res and "supersedes" in res["error"]
    assert "replace" in res["error"] or "edit_memory" in res["error"]
    assert not (memory._tenant_dir("t1") / "open_loops.md").exists(), "при отказе файл не пишется"


async def test_recall_finds_loop_by_body_word_and_returns_it_whole(tmp_path):
    """recall находит петлю по слову из тела и отдаёт целиком (превью — только у docs/)."""
    await tools.dispatch(
        "save_memory",
        {"type": "loop", "slug": "x5-presentation", "status": "open",
         "content": "Согласовать слайды для ритейлера X5 к августу",
         "source": "сессия 2026-07"}, TENANT)
    hits = await memory.recall(TENANT, "слайды")
    loops = [h for h in hits if h.get("path", "").startswith("loops/")]
    assert loops, "петля должна находиться по слову из тела"
    hit = loops[0]
    assert "doc" not in hit, "петля — не документ, превью ей не положено"
    assert "Согласовать слайды для ритейлера X5 к августу" in hit["content"], "отдаётся целиком"
    assert "_status: open_" in hit["content"], "статус приезжает в recall вместе с телом"

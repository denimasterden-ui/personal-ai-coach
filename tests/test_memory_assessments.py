"""memory.py — тесты и исходные документы: структурированный результат теста
(type=test) хранится отдельно, исходный документ (type=doc) — отдельно и не
затапливает recall, но остаётся доступен целиком через load_doc. Старые
каталоги читаются без ошибок."""
import memory


def _set_tenants(tmp_path, monkeypatch):
    monkeypatch.setattr(memory.config, "TENANTS_DIR", tmp_path / "tenants")
    (tmp_path / "tenants").mkdir()
    return "u7"


async def test_save_and_recall_test_result(tmp_path, monkeypatch):
    """Критерий 1: результат теста сохраняется как отдельный вид памяти, виден
    инструмент + дата + расклад, находится поиском."""
    tenant = _set_tenants(tmp_path, monkeypatch)
    await memory.save_memory(
        tenant, "test",
        "Инструмент: CliftonStrengths\nПройден: 2024-03-10\nРасклад: Strategic 5, Achiever 9",
        slug="clifton-2024", source="CliftonStrengths, отчёт 2024-03")
    hits = await memory.recall(tenant, "CliftonStrengths")
    assert hits, "результат теста должен находиться в recall"
    blob = "\n".join(h["content"] for h in hits)
    assert "CliftonStrengths" in blob
    assert "Strategic" in blob
    assert "2024-03-10" in blob


async def test_save_doc_and_load_whole(tmp_path, monkeypatch):
    """Критерий 2: документ сохраняется и возвращается целиком по slug."""
    tenant = _set_tenants(tmp_path, monkeypatch)
    body = "\n".join(f"строка {i} отчёта" for i in range(200))
    await memory.save_memory(tenant, "doc", body, slug="hogan-report",
                             source="Hogan, отчёт 2024-02 (PDF→текст)")
    loaded = await memory.load_doc(tenant, "hogan-report")
    assert loaded is not None
    assert loaded.count("строка") == 200


async def test_big_doc_not_in_recall_full_but_discoverable(tmp_path, monkeypatch):
    """Критерий 3: большой документ не приходит полным текстом в recall, но
    из выдачи понятно, что он существует и как его запросить."""
    tenant = _set_tenants(tmp_path, monkeypatch)
    body = "\n".join(f"Hogan Hogan Hogan строка {i}" for i in range(5000))
    await memory.save_memory(tenant, "doc", body, slug="hogan-big",
                             source="Hogan, отчёт 2024")
    hits = await memory.recall(tenant, "Hogan")
    assert hits, "документ должен обнаруживаться в recall"
    hit = next(h for h in hits if h.get("path", "").startswith("docs"))
    # полный текст (5000 повторов строки) не должен попасть в recall
    assert hit["content"].count("строка") < 5000
    # но должна быть подсказка, как получить полный текст
    assert 'load_doc("hogan-big")' in hit["content"]
    assert hit["content"].count("Hogan") > 0, "превью должно намекать на содержимое"


async def test_recall_returns_full_text_for_non_doc(tmp_path, monkeypatch):
    """Регрессия: для не-doc типов recall по-прежнему отдаёт полный текст."""
    tenant = _set_tenants(tmp_path, monkeypatch)
    body = "\n".join(f"паттерн строка {i}" for i in range(50))
    await memory.save_memory(tenant, "pattern", body, slug="p",
                             source="наблюдение в сессии")
    hits = await memory.recall(tenant, "паттерн")
    assert hits
    assert hits[0]["content"].count("строка") == 50


async def test_memory_view_counts_docs_and_tests(tmp_path, monkeypatch):
    """Критерий 4: /memory не выводит сырой текст документов, но показывает
    их количество вместе с тестами."""
    tenant = _set_tenants(tmp_path, monkeypatch)
    await memory.save_memory(tenant, "test", "CliftonStrengths ...", slug="t1",
                             source="тест 2024")
    await memory.save_memory(tenant, "doc", "большой отчёт " * 1000, slug="d1",
                             source="отчёт 2024")
    view = await memory.summarize(tenant)
    assert "1 результатов тестов" in view
    assert "1 документов" in view
    # сырой текст документа в /memory не попадает
    assert "большой отчёт" not in view


async def test_export_includes_docs(tmp_path, monkeypatch):
    """Критерий 5: /export отдаёт документы вместе с остальной памятью."""
    tenant = _set_tenants(tmp_path, monkeypatch)
    await memory.save_memory(tenant, "doc", "содержимое отчёта Хогана целиком",
                             slug="hogan", source="Hogan 2024")
    dump = await memory.export_tenant(tenant)
    assert "содержимое отчёта Хогана целиком" in dump


async def test_recall_works_on_old_catalog(tmp_path, monkeypatch):
    """Критерий 6/7: старый каталог (без tests/ и docs/) читается recall."""
    tenant = _set_tenants(tmp_path, monkeypatch)
    d = memory._tenant_dir(tenant)
    d.mkdir(parents=True, exist_ok=True)
    memory._write_text(d / "self.md", "профиль из старого каталога\n\n_updated: 2023-01-01T00:00:00+00:00_\n")
    hits = await memory.recall(tenant, "профиль")
    assert hits
    assert "профиль" in hits[0]["content"].lower()


async def test_memory_view_works_on_old_catalog(tmp_path, monkeypatch):
    """Критерий 6/7: старый каталог читается /memory без ошибок."""
    tenant = _set_tenants(tmp_path, monkeypatch)
    d = memory._tenant_dir(tenant)
    d.mkdir(parents=True, exist_ok=True)
    memory._write_text(d / "self.md", "старый профиль\n\n_updated: 2023-01-01T00:00:00+00:00_\n")
    view = await memory.summarize(tenant)
    assert "старый профиль" in view


async def test_dispatch_test_requires_source(tmp_path, monkeypatch):
    """Гейт провенанса обязателен и для новых видов записей — не обходится."""
    import tools
    monkeypatch.setattr(memory.config, "TENANTS_DIR", tmp_path)
    res = await tools.dispatch(
        "save_memory", {"type": "test", "content": "результат", "slug": "t"}, "t1")
    assert "error" in res and "source" in res["error"]
    assert not list(tmp_path.rglob("*.md"))


async def test_dispatch_load_doc(tmp_path, monkeypatch):
    """dispatch(load_doc) возвращает полный текст документа по slug."""
    import tools
    monkeypatch.setattr(memory.config, "TENANTS_DIR", tmp_path)
    await memory.save_memory(tenant := "t1", "doc", "полный текст отчёта",
                             slug="rep", source="отчёт 2024")
    res = await tools.dispatch("load_doc", {"slug": "rep"}, tenant)
    assert "полный текст отчёта" in res["content"]


async def test_load_doc_missing_returns_error(tmp_path, monkeypatch):
    import tools
    monkeypatch.setattr(memory.config, "TENANTS_DIR", tmp_path)
    res = await tools.dispatch("load_doc", {"slug": "nope"}, "t1")
    assert "error" in res


async def test_slugless_test_or_doc_is_refused(tmp_path, monkeypatch):
    """_resolve falls back to a shared "entry" filename, so a second slugless
    save overwrites the first and still reports success. Two uploaded reports
    would silently become one."""
    import tools
    monkeypatch.setattr(memory.config, "TENANTS_DIR", tmp_path)
    for kind in ("test", "doc"):
        res = await tools.dispatch(
            "save_memory", {"type": kind, "content": "x", "source": "отчёт"}, "t1")
        assert "error" in res and "slug" in res["error"], kind
    assert not list(tmp_path.rglob("*.md"))


async def test_two_reports_with_slugs_both_survive(tmp_path, monkeypatch):
    """The positive case the refusal exists to protect."""
    import tools
    monkeypatch.setattr(memory.config, "TENANTS_DIR", tmp_path)
    for slug, body in (("hogan-2024", "отчёт Хогана"), ("clifton-2021", "отчёт Гэллап")):
        await tools.dispatch("save_memory",
                             {"type": "doc", "content": body, "slug": slug,
                              "source": f"файл {slug}"}, "t1")
    assert "отчёт Хогана" in await memory.load_doc("t1", "hogan-2024")
    assert "отчёт Гэллап" in await memory.load_doc("t1", "clifton-2021")

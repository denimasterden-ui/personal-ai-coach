"""memory.py — surgical edits.

Single-file memory could only append or be rewritten whole. Correcting one wrong
claim in a 47KB profile meant leaving the wrong line above the correction (the
model then reads both and can't tell which wins) or regenerating everything.
This is the Edit-tool contract: exact fragment, unique match, loud failure.
"""
import memory
import tools


def _tenant(tmp_path, monkeypatch):
    monkeypatch.setattr(memory.config, "TENANTS_DIR", tmp_path / "tenants")
    (tmp_path / "tenants").mkdir()
    return "ed"


async def test_edit_replaces_only_the_named_fragment(tmp_path, monkeypatch):
    t = _tenant(tmp_path, monkeypatch)
    await memory.save_memory(t, "self", "## 2\nCliftonStrengths: Arranger.\n\n## 3\nЦенности.",
                             source="профиль при онбординге")
    res = await memory.edit_memory(t, "self", "CliftonStrengths: Arranger.",
                                   "CliftonStrengths: Command.", source="отчёт Gallup 2021")
    assert "edited" in res
    text = memory._read_text(memory._tenant_dir(t) / "self.md")
    assert "Command" in text and "Arranger" not in text
    assert "## 3\nЦенности." in text, "the rest of the file must survive untouched"


async def test_edit_records_where_the_correction_came_from(tmp_path, monkeypatch):
    t = _tenant(tmp_path, monkeypatch)
    await memory.save_memory(t, "self", "неверно", source="онбординг")
    await memory.edit_memory(t, "self", "неверно", "верно", source="отчёт Gallup 2021")
    text = memory._read_text(memory._tenant_dir(t) / "self.md")
    assert "отчёт Gallup 2021" in text


async def test_edit_refuses_an_ambiguous_fragment(tmp_path, monkeypatch):
    """Two matches mean the model can't know which place it's changing."""
    t = _tenant(tmp_path, monkeypatch)
    await memory.save_memory(t, "self", "строка\nдругое\nстрока", source="онбординг")
    res = await memory.edit_memory(t, "self", "строка", "иная", source="вывод")
    assert "error" in res and "2 раз" in res["error"]
    assert "иная" not in memory._read_text(memory._tenant_dir(t) / "self.md")


async def test_edit_refuses_a_fragment_that_is_not_there(tmp_path, monkeypatch):
    t = _tenant(tmp_path, monkeypatch)
    await memory.save_memory(t, "self", "что-то", source="онбординг")
    res = await memory.edit_memory(t, "self", "чего тут нет", "новое", source="вывод")
    assert "error" in res


async def test_edit_of_a_missing_record_is_an_error_not_a_new_file(tmp_path, monkeypatch):
    t = _tenant(tmp_path, monkeypatch)
    res = await memory.edit_memory(t, "pattern", "a", "b", slug="нет-такого", source="вывод")
    assert "error" in res
    assert not list((tmp_path / "tenants").rglob("*.md"))


async def test_footer_does_not_pile_up_across_edits(tmp_path, monkeypatch):
    t = _tenant(tmp_path, monkeypatch)
    await memory.save_memory(t, "self", "версия1", source="онбординг")
    await memory.edit_memory(t, "self", "версия1", "версия2", source="правка1")
    await memory.edit_memory(t, "self", "версия2", "версия3", source="правка2")
    text = memory._read_text(memory._tenant_dir(t) / "self.md")
    assert text.count("_updated:") == 1, "one footer, not a growing changelog"
    assert "правка2" in text


async def test_gate_requires_a_source_for_an_edit(tmp_path, monkeypatch):
    monkeypatch.setattr(memory.config, "TENANTS_DIR", tmp_path)
    res = await tools.dispatch(
        "edit_memory", {"type": "self", "old_string": "a", "new_string": "b"}, "t1")
    assert "error" in res and "source" in res["error"]


async def test_raw_documents_are_not_editable(tmp_path, monkeypatch):
    """A doc is the evidence other claims cite — editing it rewrites history."""
    monkeypatch.setattr(memory.config, "TENANTS_DIR", tmp_path)
    await memory.save_memory("t1", "doc", "исходный отчёт", slug="rep", source="файл")
    res = await tools.dispatch(
        "edit_memory", {"type": "doc", "old_string": "исходный", "new_string": "правленый",
                        "slug": "rep", "source": "вывод"}, "t1")
    assert "error" in res
    assert "исходный отчёт" in await memory.load_doc("t1", "rep")

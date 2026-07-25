"""memory.py — provenance: a saved assertion must carry where it came from.

These live apart from the assessment tests on purpose. The provenance line is
the whole point of the feature and it is invisible in behaviour — nothing else
breaks when it silently stops being written — so it needs tests that fail loudly
and cannot be swept away while someone edits an adjacent concern.
"""
import memory
import tools


def _tenant(tmp_path, monkeypatch):
    monkeypatch.setattr(memory.config, "TENANTS_DIR", tmp_path / "tenants")
    (tmp_path / "tenants").mkdir()
    return "prov"


async def test_source_is_written_into_the_file(tmp_path, monkeypatch):
    """The assertion and its origin must sit together on disk — a profile claim
    that lost its source is indistinguishable from a typo someone seeded."""
    t = _tenant(tmp_path, monkeypatch)
    await memory.save_memory(t, "self", "Individualization #1",
                             source="Gallup CliftonStrengths 34, 2021-12-13")
    text = memory._read_text(memory._tenant_dir(t) / "self.md")
    assert "_source: Gallup CliftonStrengths 34, 2021-12-13_" in text


async def test_multiline_source_is_flattened(tmp_path, monkeypatch):
    """A source with newlines must not break the footer it lives in."""
    t = _tenant(tmp_path, monkeypatch)
    await memory.save_memory(t, "self", "утверждение", source="отчёт\nHogan\n2024")
    text = memory._read_text(memory._tenant_dir(t) / "self.md")
    assert "_source: отчёт Hogan 2024_" in text


async def test_no_source_keeps_the_pre_provenance_file_shape(tmp_path, monkeypatch):
    """Old catalogues were written before provenance existed; the writer must
    still be able to produce that exact shape or restores break."""
    t = _tenant(tmp_path, monkeypatch)
    await memory.save_memory(t, "self", "утверждение", source=None)
    text = memory._read_text(memory._tenant_dir(t) / "self.md")
    assert "_source:" not in text
    assert "_updated:" in text


async def test_source_survives_into_the_memory_view(tmp_path, monkeypatch):
    """/memory strips the _updated footer; it must not strip the source too —
    the user has to be able to see what a claim about them rests on."""
    t = _tenant(tmp_path, monkeypatch)
    await memory.save_memory(t, "self", "Individualization #1",
                             source="Gallup CliftonStrengths 34, 2021-12-13")
    view = await memory.summarize(t)
    assert "Gallup CliftonStrengths 34" in view


async def test_appended_entries_each_keep_their_own_source(tmp_path, monkeypatch):
    """self.md grows by append — a second claim from a different origin must not
    inherit or overwrite the first one's source."""
    t = _tenant(tmp_path, monkeypatch)
    await memory.save_memory(t, "self", "первое", source="тест Gallup 2021")
    await memory.save_memory(t, "self", "второе", source="слова в сессии 2026-07")
    text = memory._read_text(memory._tenant_dir(t) / "self.md")
    assert "_source: тест Gallup 2021_" in text
    assert "_source: слова в сессии 2026-07_" in text


async def test_gate_refuses_a_sourceless_write(tmp_path, monkeypatch):
    """A schema description is advisory and gets skipped under load. The refusal
    is the actual mechanism — nothing may reach disk without a declared origin."""
    monkeypatch.setattr(memory.config, "TENANTS_DIR", tmp_path)
    res = await tools.dispatch("save_memory", {"type": "self", "content": "утверждение"}, "t1")
    assert "error" in res and "source" in res["error"]
    assert not list(tmp_path.rglob("*.md"))


async def test_gate_refuses_a_blank_source(tmp_path, monkeypatch):
    """Whitespace must not satisfy the gate — that is the cheapest way around it."""
    monkeypatch.setattr(memory.config, "TENANTS_DIR", tmp_path)
    res = await tools.dispatch(
        "save_memory", {"type": "self", "content": "утверждение", "source": "   "}, "t1")
    assert "error" in res
    assert not list(tmp_path.rglob("*.md"))


async def test_gate_lets_a_sourced_write_through(tmp_path, monkeypatch):
    """The positive case: a gate proven only on refusals may reject everyone."""
    monkeypatch.setattr(memory.config, "TENANTS_DIR", tmp_path)
    res = await tools.dispatch(
        "save_memory",
        {"type": "self", "content": "утверждение", "source": "моя гипотеза"}, "t1")
    assert "error" not in res
    assert list(tmp_path.rglob("*.md")), "a sourced write must land"

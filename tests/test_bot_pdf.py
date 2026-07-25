"""PDF attachments: routing in _handle and the fail-safe in _pdf_text.

The point of these tests is the boundary, not pypdf: an attached file must land
in the turn buffer as ordinary material and nowhere else. Every rejection path
must still say something back — the bug that started this was a message type
that produced total silence.
"""
import pytest

import bot
import memory


@pytest.fixture(autouse=True)
def isolated_tenants(tmp_path, monkeypatch):
    """_handle now writes the document into memory for real. Without this the
    suite creates directories under the live tenants/ dir — harmless here, a
    collision with real user data anywhere else."""
    monkeypatch.setattr(memory.config, "TENANTS_DIR", tmp_path / "tenants")
    (tmp_path / "tenants").mkdir()


@pytest.fixture
def tg(monkeypatch):
    """Capture outbound Telegram calls; neuter access gating and analytics."""
    sent = []

    async def fake_tg(client, method, **params):
        sent.append((method, params))
        if method == "getFile":
            return {"result": {"file_path": "documents/file.pdf"}}
        return {"ok": True}

    monkeypatch.setattr(bot, "_tg", fake_tg)
    monkeypatch.setattr(bot, "PUBLIC_MODE", True)
    monkeypatch.setattr(bot, "_rate_limited", lambda _: False)
    monkeypatch.setattr(bot.analytics, "has_seen", lambda _: True)
    monkeypatch.setattr(bot.analytics, "log", lambda *a, **k: None)
    return sent


@pytest.fixture
def buffered(monkeypatch):
    """Capture what reaches the turn buffer — the only place a file may land."""
    out = []
    monkeypatch.setattr(bot, "_buffer", lambda c, cid, text, **kw: out.append(text))
    return out


class FakeClient:
    def __init__(self, payload=b"%PDF-fake"):
        self.payload = payload

    async def get(self, url, **kw):
        class R:
            content = self.payload
        return R()


def _msg(**doc):
    d = {"file_id": "f1", "file_name": "report.pdf", "mime_type": "application/pdf"}
    d.update(doc)
    return {"chat": {"id": 7}, "message_id": 1, "document": d}


def _texts(sent):
    return " ".join(p.get("text", "") for m, p in sent if m == "sendMessage")


async def test_pdf_text_lands_in_the_turn_buffer(tg, buffered, monkeypatch):
    monkeypatch.setattr(bot, "_pdf_text", _const("Индивидуализация, Распорядитель"))
    await bot._handle(FakeClient(), _msg())
    assert len(buffered) == 1
    assert "Индивидуализация, Распорядитель" in buffered[0]
    assert "report.pdf" in buffered[0], "coach must know this was an attachment"


async def test_long_pdf_says_the_turn_holds_only_the_start(tg, buffered, monkeypatch):
    """The turn carries a preview, so both the user and the coach must be told
    so — reasoning over a silent stub as if it were whole is the failure that
    started this."""
    monkeypatch.setattr(bot, "_pdf_text", _const("я" * (bot.PDF_MAX_CHARS + 500)))
    await bot._handle(FakeClient(), _msg())
    assert len(buffered[0]) <= bot.PDF_MAX_CHARS + 300  # + the header
    assert str(bot.PDF_MAX_CHARS) in _texts(tg), "user must learn the turn is partial"
    assert str(bot.PDF_MAX_CHARS) in buffered[0], "coach must learn the turn is partial"


async def test_scan_without_text_layer_is_refused_not_silent(tg, buffered, monkeypatch):
    monkeypatch.setattr(bot, "_pdf_text", _const(""))
    await bot._handle(FakeClient(), _msg())
    assert buffered == []
    assert "скан" in _texts(tg)


async def test_non_pdf_document_is_refused_without_downloading(tg, buffered):
    await bot._handle(FakeClient(), _msg(file_name="notes.docx", mime_type="application/msword"))
    assert buffered == []
    assert "PDF" in _texts(tg)
    assert not any(m == "getFile" for m, _ in tg), "must not fetch what it won't read"


async def test_oversize_pdf_is_refused_without_downloading(tg, buffered):
    await bot._handle(FakeClient(), _msg(file_size=bot.PDF_MAX_BYTES + 1))
    assert buffered == []
    assert "20 МБ" in _texts(tg)
    assert not any(m == "getFile" for m, _ in tg)


async def test_getfile_failure_answers_instead_of_hanging(tg, buffered, monkeypatch):
    async def failing_tg(client, method, **params):
        tg.append((method, params))
        return {"ok": False, "description": "file is too big"}
    monkeypatch.setattr(bot, "_tg", failing_tg)
    await bot._handle(FakeClient(), _msg())
    assert buffered == []
    assert "ещё раз" in _texts(tg)


async def test_unsupported_type_gets_an_answer(tg, buffered):
    """The original defect: a photo/sticker was logged and silently dropped."""
    await bot._handle(FakeClient(), {"chat": {"id": 7}, "message_id": 1, "photo": [{}]})
    assert buffered == []
    assert _texts(tg), "unsupported input must never be met with silence"


async def test_pdf_text_never_raises_on_garbage():
    """Fail-safe: a corrupt or encrypted file must degrade to 'no text layer'."""
    assert await bot._pdf_text(b"this is not a pdf at all") == ""


def _const(value):
    async def _f(_data):
        return value
    return _f


async def test_whole_document_is_stored_not_just_the_preview(tg, buffered, monkeypatch):
    """The point of the change: the clipped part must survive somewhere. A
    55k-char assessment report losing half of itself was the original defect."""
    body = "я" * (bot.PDF_MAX_CHARS + 5000)
    monkeypatch.setattr(bot, "_pdf_text", _const(body))
    await bot._handle(FakeClient(), _msg())
    stored = await memory.load_doc(bot._tenant_id_for(7), memory._slug("report.pdf"))
    assert stored is not None
    assert body in stored, "document must be kept whole, not just the preview"


async def test_turn_tells_the_coach_how_to_read_the_rest(tg, buffered, monkeypatch):
    monkeypatch.setattr(bot, "_pdf_text", _const("текст отчёта"))
    await bot._handle(FakeClient(), _msg())
    assert 'load_doc("report-pdf")' in buffered[0]


async def test_stored_document_carries_its_provenance(tg, buffered, monkeypatch):
    """A document is a source for profile claims, so it must say what it is."""
    monkeypatch.setattr(bot, "_pdf_text", _const("текст отчёта"))
    await bot._handle(FakeClient(), _msg())
    d = memory._tenant_dir(bot._tenant_id_for(7))
    text = memory._read_text(d / "docs" / "report-pdf.md")
    assert "_source:" in text and "report.pdf" in text


async def test_failed_save_is_not_reported_as_success(tg, buffered, monkeypatch):
    """save_memory returns {'error': ...} instead of raising. Claiming success
    anyway would hand the coach a load_doc slug that resolves to nothing."""
    async def failing_save(*a, **k):
        return {"error": "disk gone"}
    monkeypatch.setattr(bot, "_pdf_text", _const("текст отчёта"))
    monkeypatch.setattr(memory, "save_memory", failing_save)
    await bot._handle(FakeClient(), _msg())
    assert "не смог" in _texts(tg)
    assert "load_doc" not in buffered[0], "must not advertise a document that isn't there"

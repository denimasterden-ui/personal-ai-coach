"""PDF attachments: routing in _handle and the fail-safe in _pdf_text.

The point of these tests is the boundary, not pypdf: an attached file must land
in the turn buffer as ordinary material and nowhere else. Every rejection path
must still say something back — the bug that started this was a message type
that produced total silence.
"""
import pytest

import bot


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


async def test_long_pdf_is_clipped_and_the_user_is_told(tg, buffered, monkeypatch):
    monkeypatch.setattr(bot, "_pdf_text", _const("я" * (bot.PDF_MAX_CHARS + 500)))
    await bot._handle(FakeClient(), _msg())
    assert len(buffered[0]) <= bot.PDF_MAX_CHARS + 200  # + the header
    assert "обрезал" in _texts(tg)


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

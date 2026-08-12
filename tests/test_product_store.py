"""Служебное хранилище продуктовой обвязки (aicoach-dev#14, ADR 0004).

chat_id каждого, кто нажал /start, сохраняется в отдельном зашифрованном файле
— тот же ключ и паттерн, что у подписок на еженедельные напоминания. Хэш
тенанта односторонний: без этого хранилища ушедший человек недостижим навсегда.
"""
import json

import pytest
from cryptography.fernet import Fernet

import bot


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    """Point the product store at a temp file; start each test with a clean dict."""
    store = tmp_path / "contacts.enc"
    monkeypatch.setattr(bot, "PRODUCT_STORE_FILE", store)
    bot._contacts.clear()


@pytest.fixture
def tg(monkeypatch):
    """Capture outbound Telegram calls; neuter access gating and analytics."""
    sent = []

    async def fake_tg(client, method, **params):
        sent.append((method, params))
        return {"ok": True}

    monkeypatch.setattr(bot, "_tg", fake_tg)
    monkeypatch.setattr(bot, "PUBLIC_MODE", True)
    monkeypatch.setattr(bot, "_rate_limited", lambda _: False)
    monkeypatch.setattr(bot.analytics, "has_seen", lambda _: True)
    monkeypatch.setattr(bot.analytics, "log", lambda *a, **k: None)
    return sent


class FakeClient:
    """Minimal stand-in: /delete_confirm calls client.delete for session reset."""

    async def delete(self, url, **kw):
        return None


async def test_start_saves_chat_id(tg):
    """/start is the first contact — chat_id must land in the product store."""
    await bot._command(FakeClient(), 42, "/start")

    assert "42" in bot._contacts, "chat_id must be saved on /start"
    assert bot.PRODUCT_STORE_FILE.exists(), "store must be persisted to disk"


async def test_start_invites_material_and_keeps_the_disclaimer(tg):
    """The Telegram seam keeps the essential first-contact promises visible.
    The privacy detail deliberately isn't here — a wall of caveats on screen one
    is what was scaring people off — but the way to it must be."""
    await bot._command(FakeClient(), 42, "/start")

    text = next(params["text"] for method, params in tg if method == "sendMessage")
    assert "тест, опросник либо стенограмму" in text
    assert "не заменяю психотерапевта" in text
    assert "/privacy" in text


async def test_privacy_notice_states_what_is_true_of_the_code(tg):
    """Layered notice: the detail lives a tap away. Every claim in it has to
    match the code — a reassurance the practice contradicts is worse than none.
    The operator really does read разборы through supervise.py, so it says so."""
    await bot._command(FakeClient(), 42, "/privacy")

    text = " ".join(params["text"] for method, params in tg if method == "sendMessage")
    assert "просматриваю" in text, "operator review must be disclosed, not implied away"
    assert "/delete_my_data" in text
    assert "оператора" in text, "the key lives on the same server — say it"


async def test_chat_id_survives_reload(tg):
    """A bot restart (RAM cleared) must not lose the chat_id — reload from disk."""
    await bot._command(FakeClient(), 42, "/start")
    assert "42" in bot._contacts

    # Simulate a restart: in-RAM dict wiped, then _load_contacts repopulates it.
    bot._contacts.clear()
    assert not bot._contacts, "precondition: RAM cleared"

    bot._load_contacts()
    assert "42" in bot._contacts, "chat_id must survive reload from disk"


async def test_delete_confirm_removes_chat_id(tg, monkeypatch):
    """/delete_confirm must purge the person from the product store, not just memory."""
    await bot._command(FakeClient(), 42, "/start")
    assert "42" in bot._contacts

    async def fake_delete(tenant_id):
        return True
    monkeypatch.setattr(bot.memory, "delete_tenant", fake_delete)

    await bot._command(FakeClient(), 42, "/delete_confirm")

    assert "42" not in bot._contacts, "chat_id must be removed on /delete_confirm"
    # Verify the removal was persisted — reload from disk and check again.
    bot._contacts.clear()
    bot._load_contacts()
    assert "42" not in bot._contacts, "removal must be persisted to disk"


async def test_delete_confirm_reaches_the_quality_layer(tg, monkeypatch):
    """supervision.db holds the ход verbatim too. Deleting only tenants/ left the
    conversation sitting there while the bot said «вся твоя память удалена»."""
    forgotten = []
    monkeypatch.setattr(bot.supervision, "forget_tenant", forgotten.append)

    async def fake_delete(tenant_id):
        return True
    monkeypatch.setattr(bot.memory, "delete_tenant", fake_delete)

    await bot._command(FakeClient(), 42, "/delete_confirm")

    assert forgotten == [bot._tenant_id_for(42)]


async def test_store_is_encrypted_on_disk(tg, monkeypatch):
    """Raw file bytes must not contain a plaintext chat_id — same threat model as
    subscribers.enc: a leaked file must not read as a Telegram roster."""
    key = Fernet.generate_key()
    monkeypatch.setattr(bot, "_sub_fernet", Fernet(key))

    await bot._command(FakeClient(), 999888, "/start")

    raw = bot.PRODUCT_STORE_FILE.read_bytes()
    assert b"999888" not in raw, "chat_id must not appear in plaintext on disk"
    # And decryption must round-trip the data back.
    decrypted = Fernet(key).decrypt(raw)
    assert "999888" in json.loads(decrypted)


async def test_encrypted_store_loads_after_restart(tg, monkeypatch):
    """The Fernet-encrypted store must reload correctly across a restart."""
    key = Fernet.generate_key()
    monkeypatch.setattr(bot, "_sub_fernet", Fernet(key))

    await bot._command(FakeClient(), 555, "/start")
    bot._contacts.clear()
    bot._load_contacts()
    assert "555" in bot._contacts, "encrypted store must reload correctly"


async def test_corrupt_store_does_not_crash(tg):
    """A corrupt or unreadable store file must not crash the bot at startup —
    it logs and continues with an empty dict, same fail-safe as _load_subs."""
    bot.PRODUCT_STORE_FILE.write_bytes(b"\x00\x01\x02 not json and not encrypted \xff")

    # Must not raise.
    bot._load_contacts()
    assert bot._contacts == {}, "corrupt store leaves contacts empty, bot stays up"


async def test_corrupt_encrypted_store_does_not_crash(tg, monkeypatch):
    """Same, but with a Fernet key set — a truncated/corrupted ciphertext is a
    different failure path (InvalidToken), and must not crash either."""
    key = Fernet.generate_key()
    monkeypatch.setattr(bot, "_sub_fernet", Fernet(key))
    bot.PRODUCT_STORE_FILE.write_bytes(b"gAAAAABm_corrupted_token_here====")

    bot._load_contacts()
    assert bot._contacts == {}, "corrupt encrypted store leaves contacts empty"


async def test_store_is_separate_from_subscribers(tg, monkeypatch, tmp_path):
    """The product store must not be mixed into subscribers.enc — separate file,
    separate format (ticket: 'отдельный файл рядом, не смешивать')."""
    subs_file = tmp_path / "subscribers.enc"
    monkeypatch.setattr(bot, "SUBSCRIBERS_FILE", subs_file)

    await bot._command(FakeClient(), 42, "/start")

    assert bot.PRODUCT_STORE_FILE.exists(), "contacts written to their own file"
    assert not subs_file.exists(), "contacts must not leak into subscribers file"


async def test_repeated_start_preserves_tour_state(tg):
    """A returning user pressing /start again must not clobber tour progress —
    the store value reserves room for tour state (ticket: 'место под состояние
    прохождения тура'), and a re-onboarding is not a reset."""
    bot._contacts["77"] = {"tour": "done"}
    bot._save_contacts()

    await bot._command(FakeClient(), 77, "/start")

    assert bot._contacts["77"] == {"tour": "done"}, "tour state must survive re-/start"

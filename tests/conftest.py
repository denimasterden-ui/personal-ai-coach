"""Shared test fixtures. Keep tests hermetic: no real network, no real secrets,
no real LLM. Env vars some modules read at import time are stubbed here."""
import os

import pytest

os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("TENANT_SALT", "test-salt")
os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("TG_BOT_TOKEN", "test:token")


@pytest.fixture(autouse=True)
def _clear_service_sessions():
    """service._sessions is a module-level dict — a test that leaves a session
    in it leaks into whichever test runs next (surfaced by aicoach-dev#13+#17
    landing together: test_portrait.py asserts an empty dict and inherited a
    stray "same" session from test_first_contact.py). Clear on both sides so
    order never matters."""
    import service
    service._sessions.clear()
    yield
    service._sessions.clear()


@pytest.fixture(autouse=True)
def _clear_bot_contacts():
    """bot._contacts is a module-level dict too — the first-contact marker
    (aicoach-dev#17, moved out of tenant memory into the product store to
    honour ADR 0004) is written by the real, unmocked code path in several
    bot tests, so a leftover chat_id entry would leak between test files."""
    import bot
    bot._contacts.clear()
    yield
    bot._contacts.clear()

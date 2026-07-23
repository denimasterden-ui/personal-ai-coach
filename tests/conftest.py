"""Shared test fixtures. Keep tests hermetic: no real network, no real secrets,
no real LLM. Env vars some modules read at import time are stubbed here."""
import os

os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("TENANT_SALT", "test-salt")
os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("TG_BOT_TOKEN", "test:token")

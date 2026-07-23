"""analytics.py — retention math and fail-safe logging.

Covers TEST_PLAN.md rows: test_retention_math, test_analytics_safe.
Retention is the easiest thing to get subtly wrong (cohort day boundaries),
so pin it; and analytics must never raise into the coaching flow.
"""
import time

import pytest

import analytics


@pytest.fixture
def db(tmp_path, monkeypatch):
    """Isolated analytics DB per test."""
    monkeypatch.setattr(analytics, "DB_FILE", tmp_path / "test.db")
    analytics.init()
    return analytics


def _seed(day, tenant, event, **kw):
    """Insert one event stamped on an exact UTC day-index."""
    ts = day * 86400 + 100
    with analytics._conn() as c:
        c.execute(
            "INSERT INTO events (ts, day, tenant, event, kind, cmd, mtype, source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (ts, day, tenant, event, kw.get("kind"), kw.get("cmd"),
             kw.get("mtype"), kw.get("source")),
        )


def test_log_never_raises_into_flow(db, monkeypatch):
    """analytics.log swallows errors — coaching flow must not break on a bad write."""
    def boom():
        raise RuntimeError("db exploded")
    monkeypatch.setattr(analytics, "_conn", boom)
    # must NOT raise
    analytics.log("message", "tenant-x", kind="voice")


def test_d1_retention_counts_returners(db):
    today = analytics._today()
    # A: first_seen D-3, returns on D-2 (D+1) -> retained
    _seed(today - 3, "A", "first_seen")
    _seed(today - 2, "A", "message")
    # B: first_seen D-3, never returns -> not retained
    _seed(today - 3, "B", "first_seen")

    s = analytics.summary(days=30)
    assert s["d1_retention"] == 50  # 1 of 2


def test_activation_share(db):
    today = analytics._today()
    _seed(today - 1, "A", "first_seen")
    _seed(today - 1, "A", "message")   # activated
    _seed(today - 1, "B", "first_seen")  # never messaged
    s = analytics.summary(days=30)
    assert s["activation_pct"] == 50


def test_dau_counts_distinct_today(db):
    today = analytics._today()
    _seed(today, "A", "message")
    _seed(today, "A", "message")  # same tenant twice -> still 1
    _seed(today, "B", "message")
    s = analytics.summary(days=30)
    assert s["dau"] == 2


def test_empty_db_is_safe(db):
    s = analytics.summary(days=30)
    assert s["dau"] == 0
    assert s["d1_retention"] is None  # no cohorts -> None, not a crash
    # format must render without blowing up on None
    assert "AICOACH" in analytics.format_summary(s)

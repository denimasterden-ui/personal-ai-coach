"""supervision.py — capture rules, encryption round-trip, retention (ADR 0003).

The capture predicate decides what content gets stored at all, so it's the part
that must not drift: too loose and we hoard разборы we said we wouldn't, too tight
and the critic starves.
"""
import asyncio
import sqlite3
import time

import pytest

import supervision


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(supervision, "DB_FILE", tmp_path / "sup.db")
    monkeypatch.setattr(supervision, "SUPERVISED", set())
    monkeypatch.setattr(supervision, "SAMPLE_PCT", 0)  # no random control unless asked
    supervision.init()
    return supervision


LONG = "x" * 800   # substantial user input
SHORT = "y" * 100  # trivial input


def test_flagged_tenant_captures_everything(db, monkeypatch):
    monkeypatch.setattr(supervision, "SUPERVISED", {"me"})
    r = supervision.reasons_for("me", 10, 5000, ["save_memory"], False)
    assert r == ["full"], "flagged tenant must capture every turn"


def test_healthy_turn_is_not_captured(db):
    """Substantial input, long answer, memory written -> nothing to review."""
    r = supervision.reasons_for("other", len(LONG), 3000, ["recall", "save_memory"], False)
    assert r == []


def test_budget_exhaustion_is_captured(db):
    r = supervision.reasons_for("other", len(LONG), 3000, ["save_memory"], True)
    assert "budget" in r


def test_short_answer_to_substantial_input_is_captured(db):
    r = supervision.reasons_for("other", len(LONG), 200, ["save_memory"], False)
    assert "short_answer" in r


# test_no_memory_write_on_substantial_input_is_captured removed (aicoach-dev#11):
# after #10 moved memory writes to the recollect pass, the coach never calls
# save_memory, so this signal would fire on every turn. The signal now lives
# in pass_stats() instead — see test_pass_stats_counts_no_write below.


def test_trivial_input_does_not_trip_signals(db):
    """A one-liner deserves a short answer and no memory — that's not a defect."""
    r = supervision.reasons_for("other", len(SHORT), 200, [], False)
    assert r == []


def test_control_sample_can_capture_healthy_turns(db, monkeypatch):
    monkeypatch.setattr(supervision, "SAMPLE_PCT", 100)  # always sample
    r = supervision.reasons_for("other", len(LONG), 3000, ["save_memory"], False)
    assert r == ["control"]


def test_capture_roundtrip_preserves_content(db):
    supervision.capture("me2", "вопрос человека", "ответ коуча",
                        skill="Integrative Personal Coaching v1",
                        tools_used=["recall"], budget_exhausted=True)
    got = supervision.pending()
    assert len(got) == 1
    assert got[0]["user"] == "вопрос человека"
    assert got[0]["answer"] == "ответ коуча"
    assert got[0]["skill"] == "Integrative Personal Coaching v1"
    assert "budget" in got[0]["reasons"]


def test_content_is_encrypted_on_disk(db, monkeypatch):
    """Encryption is the stated mitigation for the first content store in the
    system — assert the plaintext genuinely isn't sitting in the DB file."""
    from cryptography.fernet import Fernet
    monkeypatch.setenv("MEMORY_ENCRYPTION_KEY", Fernet.generate_key().decode())
    secret = "очень личный разбор про сестру"
    supervision.capture("me2", secret, "ответ", budget_exhausted=True)

    # WAL mode keeps fresh rows in the -wal sidecar, so checking only the .db
    # file would pass even with encryption switched off.
    raw = b"".join(
        p.read_bytes()
        for p in [supervision.DB_FILE,
                  supervision.DB_FILE.with_name(supervision.DB_FILE.name + "-wal"),
                  supervision.DB_FILE.with_name(supervision.DB_FILE.name + "-shm")]
        if p.exists()
    )
    assert secret.encode() not in raw, "content stored in plaintext despite key"
    assert supervision.pending()[0]["user"] == secret  # still readable with the key


def test_capture_never_raises_into_flow(db, monkeypatch):
    def boom():
        raise RuntimeError("disk gone")
    monkeypatch.setattr(supervision, "_conn", boom)
    supervision.capture("me2", "a" * 900, "b")  # must not raise


async def test_capture_async_does_not_block_the_loop(db, monkeypatch):
    """Regression for the review blocker: capture must run off the event loop, so
    a slow disk write can't stall the answer waiting to be sent to the user."""
    started = asyncio.Event()

    def slow_capture(*a, **k):
        started.set()
        time.sleep(0.3)  # a blocking sync call, as sqlite+Fernet would be

    monkeypatch.setattr(supervision, "capture", slow_capture)

    loop = asyncio.get_running_loop()
    t0 = loop.time()
    supervision.capture_async("me2", "a" * 900, "b", budget_exhausted=True)
    # the loop stays responsive: this returns immediately, well before slow_capture ends
    await asyncio.wait_for(started.wait(), timeout=1)
    assert loop.time() - t0 < 0.2, "capture_async blocked the event loop"


def test_reviewed_cases_leave_the_queue(db):
    supervision.capture("me2", "a" * 900, "b", budget_exhausted=True)
    case = supervision.pending()[0]
    supervision.mark_reviewed(case["id"], "вердикт критика")
    assert supervision.pending() == []


def test_pending_can_scope_to_one_tenant(db):
    supervision.capture("alice", "a" * 900, "ans", budget_exhausted=True)
    supervision.capture("bob", "b" * 900, "ans", budget_exhausted=True)
    alice = supervision.pending(tenant="alice")
    assert len(alice) == 1 and alice[0]["tenant"] == "alice"
    assert len(supervision.pending()) == 2  # no filter = whole pool


def test_stats_break_down_per_tenant(db):
    supervision.capture("alice", "a" * 900, "ans", budget_exhausted=True)
    supervision.capture("alice", "a" * 900, "ans", budget_exhausted=True)
    supervision.capture("bob", "b" * 900, "ans", budget_exhausted=True)
    by_tenant = dict((t, (tot, wait)) for t, tot, wait in supervision.stats()["by_tenant"])
    assert by_tenant["alice"] == (2, 2)
    assert by_tenant["bob"] == (1, 1)


def test_stats_per_tenant_counts_only_pending_as_waiting(db):
    supervision.capture("alice", "a" * 900, "ans", budget_exhausted=True)
    supervision.capture("alice", "a" * 900, "ans", budget_exhausted=True)
    supervision.mark_reviewed(supervision.pending(tenant="alice")[0]["id"], "verdict")
    tot, wait = next((tot, w) for t, tot, w in supervision.stats()["by_tenant"] if t == "alice")
    assert (tot, wait) == (2, 1)  # 2 captured, 1 still waiting


def test_purge_drops_old_cases_only(db):
    supervision.capture("me2", "a" * 900, "b", budget_exhausted=True)
    fresh_id = supervision.pending()[0]["id"]
    # backdate one case beyond retention
    old_day = int(time.time() // 86400) - 200
    with supervision._conn() as c:
        c.execute("INSERT INTO cases (ts, day, tenant, reasons, user_len, answer_len, payload) "
                  "VALUES (?, ?, 'old', 'budget', 1, 1, ?)", (time.time(), old_day, b"{}"))
    assert supervision.purge(days=90) == 1
    assert supervision.pending()[0]["id"] == fresh_id


# ── passes (memory extraction pass trace, aicoach-dev#11) ─────────────────────


def test_log_pass_roundtrip(db):
    """log_pass writes to passes table with correct fields."""
    supervision.log_pass("alice", tool_names=["recall", "save_memory"],
                         write_count=1, write_attempts=1, recall_count=1,
                         user_text="сообщение", answer="ответ")
    with supervision._conn() as c:
        c.row_factory = sqlite3.Row
        row = c.execute("SELECT * FROM passes WHERE tenant = ?", ("alice",)).fetchone()
    assert row is not None
    assert row["write_count"] == 1
    assert row["write_attempts"] == 1
    assert row["recall_count"] == 1
    assert row["crashed"] == 0
    assert row["tool_names"] == "recall,save_memory"


def test_log_pass_crashed(db):
    """A crashed pass logs crashed=1 with error message."""
    supervision.log_pass("bob", crashed=True, error="RuntimeError: pass failed")
    with supervision._conn() as c:
        c.row_factory = sqlite3.Row
        row = c.execute("SELECT * FROM passes WHERE tenant = ?", ("bob",)).fetchone()
    assert row["crashed"] == 1
    assert "RuntimeError" in row["error"]


def test_log_pass_never_raises(db, monkeypatch):
    """log_pass swallows exceptions — a failed log must not cost a turn."""
    monkeypatch.setattr(supervision, "DB_FILE", tmp_path := __import__("pathlib").Path("/nonexistent"))
    # Should not raise
    supervision.log_pass("alice", write_count=1)


def test_pass_stats_counts_no_write(db):
    """pass_stats reports passes that ran but wrote nothing (the new signal)."""
    supervision.log_pass("alice", write_count=0, write_attempts=0)
    supervision.log_pass("alice", write_count=1, write_attempts=1)
    supervision.log_pass("bob", write_count=0, write_attempts=0)
    supervision.log_pass("bob", crashed=True, error="fail")
    stats = supervision.pass_stats(days=1)
    assert stats["total"] == 4
    assert stats["no_write"] == 2  # alice + bob (crashed doesn't count)
    assert stats["crashed"] == 1


def test_pass_stats_by_tenant(db):
    """pass_stats groups by tenant with write count."""
    supervision.log_pass("alice", write_count=1, write_attempts=1)
    supervision.log_pass("alice", write_count=0, write_attempts=0)
    supervision.log_pass("bob", write_count=2, write_attempts=2)
    stats = supervision.pass_stats(days=1)
    by_tenant = {t: (total, writes) for t, total, writes in stats["by_tenant"]}
    assert by_tenant["alice"] == (2, 1)  # 2 passes, 1 with writes
    assert by_tenant["bob"] == (1, 1)

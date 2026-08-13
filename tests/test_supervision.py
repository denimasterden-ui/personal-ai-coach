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


def _seed_established(tenant, n=3):
    """Insert n already-reviewed seed cases so the tenant counts as
    established for the onboarding rule (aicoach-dev#15). Reviewed cases
    don't pollute the pending queue."""
    now = time.time()
    with supervision._conn() as c:
        for _ in range(n):
            c.execute(
                "INSERT INTO cases (ts, day, tenant, reasons, user_len, answer_len, "
                "tools, payload, reviewed) VALUES (?, ?, ?, 'seed', 1, 1, '', '{}', 1)",
                (now, int(now // 86400), tenant),
            )


def test_flagged_tenant_captures_everything(db, monkeypatch):
    monkeypatch.setattr(supervision, "SUPERVISED", {"me"})
    r = supervision.reasons_for("me", 10, 5000, ["save_memory"], False)
    assert r == ["full"], "flagged tenant must capture every turn"


def test_healthy_turn_is_not_captured(db):
    """Substantial input, long answer, memory written -> nothing to review."""
    _seed_established("other", 3)  # established tenant, past onboarding
    r = supervision.reasons_for("other", len(LONG), 3000, ["recall", "save_memory"], False)
    assert r == []


def test_budget_exhaustion_is_captured(db):
    _seed_established("other", 3)
    r = supervision.reasons_for("other", len(LONG), 3000, ["save_memory"], True)
    assert "budget" in r


def test_short_answer_to_substantial_input_is_captured(db):
    _seed_established("other", 3)
    r = supervision.reasons_for("other", len(LONG), 200, ["save_memory"], False)
    assert "short_answer" in r


# test_no_memory_write_on_substantial_input_is_captured removed (aicoach-dev#11):
# after #10 moved memory writes to the recollect pass, the coach never calls
# save_memory, so this signal would fire on every turn. The signal now lives
# in pass_stats() instead — see test_pass_stats_counts_no_write below.


def test_trivial_input_does_not_trip_signals(db):
    """A one-liner deserves a short answer and no memory — that's not a defect."""
    _seed_established("other", 3)
    r = supervision.reasons_for("other", len(SHORT), 200, [], False)
    assert r == []


def test_control_sample_can_capture_healthy_turns(db, monkeypatch):
    monkeypatch.setattr(supervision, "SAMPLE_PCT", 100)  # always sample
    _seed_established("other", 3)
    r = supervision.reasons_for("other", len(LONG), 3000, ["save_memory"], False)
    assert r == ["control"]


def test_capture_roundtrip_preserves_content(db):
    _seed_established("me2", 3)  # established tenant, past onboarding
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


def test_purge_takes_ratings_with_the_case_they_point_at(db):
    """A rating outliving its case is unreadable — «мимо» says nothing without
    the trace beside it — and would otherwise pile up in the db forever."""
    old_ts = time.time() - 200 * 86400
    with supervision._conn() as c:
        c.execute("INSERT INTO ratings (ts, tenant, turn_id, rating) VALUES (?, 'old', 'stale', 'miss')",
                  (old_ts,))
    supervision.save_rating("me2", "fresh", "hit")

    supervision.purge(days=90)

    assert supervision.get_rating("stale") is None
    assert supervision.get_rating("fresh") == "hit"


# ── deletion: the story goes, the lesson stays ────────────────────────────────


def test_delete_erases_what_the_person_said(db):
    """The promise «стереть текст разговоров» has to hold in the quality layer
    too — this table held the ход verbatim while the bot claimed it was gone."""
    supervision.capture("alice", "у меня выгорание на работе" * 40, "ответ коуча",
                        budget_exhausted=True)
    supervision.log_pass("alice", tool_names=["recall"], recall_count=1,
                         user_text="у меня выгорание", answer="ответ коуча")

    supervision.forget_tenant("alice")

    with supervision._conn() as c:
        payloads = [r[0] for r in c.execute("SELECT payload FROM cases").fetchall()]
        payloads += [r[0] for r in c.execute("SELECT payload FROM passes").fetchall()]
    blob = b"".join(p for p in payloads if p)
    assert "выгорание".encode() not in blob
    assert blob == b"", "no trace of the conversation may survive deletion"


def test_delete_unlinks_the_rows_from_the_person(db):
    supervision.capture("alice", "a" * 900, "b", budget_exhausted=True, turn_id="t1")
    supervision.save_rating("alice", "t1", "miss")

    supervision.forget_tenant("alice")

    with supervision._conn() as c:
        tenants = {r[0] for r in c.execute("SELECT tenant FROM cases").fetchall()}
        tenants |= {r[0] for r in c.execute("SELECT tenant FROM ratings").fetchall()}
    assert tenants == {supervision.DELETED_TENANT}, "nothing may group back into a person"


def test_delete_keeps_the_rating_and_its_trace(db):
    """Denis's call: the rating is product experience, not a personal fact. It
    survives deletion — stripped of text and of any link to who gave it."""
    supervision.capture("alice", "a" * 900, "b", budget_exhausted=True,
                        turn_id="t1", tools_used=["recall"])
    supervision.save_rating("alice", "t1", "miss")

    supervision.forget_tenant("alice")

    assert supervision.get_rating("t1") == "miss"
    with supervision._conn() as c:
        tools, reasons = c.execute(
            "SELECT tools, reasons FROM cases WHERE turn_id = 't1'").fetchone()
    assert tools == "recall", "the trace that makes «мимо» diagnosable must stay"
    assert reasons


def test_delete_keeps_the_отзыв_with_its_chat_id(db):
    """Решение 14.08.2026, отдельное от оценок: отзыв о сервисе переживает
    удаление НЕ обезличенным. Это добровольное показание о продукте, а не часть
    разговора, и отзыв, который нельзя возвести к живому аккаунту, не работает
    как доказательство, что мы его не сочинили сами."""
    supervision.save_feedback("alice", 588259993, "кнопки удобные, тур длинноват")

    supervision.forget_tenant("alice")

    kept = supervision.feedback()
    assert [f["text"] for f in kept] == ["кнопки удобные, тур длинноват"]
    assert kept[0]["chat_id"] == "588259993", "связь с аккаунтом сохраняется намеренно"


def test_отзыв_is_not_on_retention(db):
    """Доказательство, которое протухает через 90 дней, — не доказательство."""
    supervision.save_feedback("alice", 1, "очень помогло")
    with supervision._conn() as c:
        c.execute("UPDATE feedback SET ts = 0")

    supervision.purge()

    assert [f["text"] for f in supervision.feedback()] == ["очень помогло"]


def test_отзыв_is_encrypted_at_rest(db, monkeypatch):
    """Свободный текст: человек напишет туда что угодно, включая своё. Лежит
    под тем же ключом, что и содержимое разборов."""
    from cryptography.fernet import Fernet
    key = Fernet.generate_key()
    monkeypatch.setenv("MEMORY_ENCRYPTION_KEY", key.decode())

    supervision.save_feedback("alice", 1, "секретное слово")

    with supervision._conn() as c:
        blob = c.execute("SELECT text FROM feedback").fetchone()[0]
    assert "секретное слово" not in blob.decode("utf-8", "ignore")
    assert [f["text"] for f in supervision.feedback()] == ["секретное слово"]


def test_delete_leaves_other_people_alone(db):
    supervision.capture("alice", "a" * 900, "b", budget_exhausted=True, turn_id="t1")
    supervision.capture("bob", "c" * 900, "d", budget_exhausted=True, turn_id="t2")
    supervision.save_rating("bob", "t2", "hit")

    supervision.forget_tenant("alice")

    bob = [c for c in supervision.pending() if c["tenant"] == "bob"]
    assert len(bob) == 1
    assert bob[0]["answer"] == "d", "another person's разбор must be untouched"
    assert supervision.get_rating("t2") == "hit"


def test_deleted_cases_leave_the_review_queue(db):
    """An emptied case has nothing left to review — left pending it would fail to
    decode on every supervision run, forever."""
    supervision.capture("alice", "a" * 900, "b", budget_exhausted=True)

    supervision.forget_tenant("alice")

    assert supervision.pending() == []


def test_delete_never_raises_into_the_flow(db, monkeypatch):
    """Deletion must not fail because the quality layer had a bad day."""
    monkeypatch.setattr(supervision, "DB_FILE", "/nonexistent/dir/sup.db")
    assert supervision.forget_tenant("alice") == 0


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


# ── onboarding capture (aicoach-dev#15) ──────────────────────────────────────
# The funnel breaks at the very first turns: most new tenants don't get past
# message one. Those turns never trip a problem signal (there's nothing wrong
# with a short first exchange), so they never enter the quality layer. The
# onboarding rule captures the first ONBOARDING_TURNS turns of a brand-new
# tenant unconditionally — "new" is self-referential: counted by how many
# cases this tenant already has stored.


def test_new_tenant_first_turn_is_captured(db):
    """A brand-new tenant's first turn is captured regardless of signals —
    trivial input, short answer, no budget issue, still captured."""
    r = supervision.reasons_for("newcomer", len(SHORT), 200, [], False)
    assert r == ["onboarding"]


def test_new_tenant_second_and_third_turns_still_captured(db):
    """With 1 and 2 stored cases the tenant is still 'new' — capture continues."""
    _seed_established("newcomer", 1)
    assert supervision.reasons_for("newcomer", len(SHORT), 200, [], False) == ["onboarding"]
    _seed_established("newcomer", 1)  # 2 total
    assert supervision.reasons_for("newcomer", len(SHORT), 200, [], False) == ["onboarding"]


def test_fourth_turn_follows_normal_rules(db):
    """After 3 stored cases the tenant is established — only signals capture."""
    _seed_established("newcomer", 3)
    r = supervision.reasons_for("newcomer", len(SHORT), 200, [], False)
    assert r == [], "no signal on turn 4+ → not captured"


def test_fourth_turn_captured_when_signal_fires(db):
    """An established tenant's turn is still captured if a signal fires."""
    _seed_established("established", 3)
    r = supervision.reasons_for("established", len(LONG), 3000, ["save_memory"], True)
    assert "budget" in r
    assert "onboarding" not in r


def test_supervised_tenant_gets_full_not_onboarding(db, monkeypatch):
    """A flagged tenant is captured with 'full'; the onboarding rule doesn't
    conflict (AC: observed tenant still captured fully)."""
    monkeypatch.setattr(supervision, "SUPERVISED", {"watched"})
    r = supervision.reasons_for("watched", len(SHORT), 200, [], False)
    assert r == ["full"]


def test_onboarding_reason_is_distinguishable(db):
    """The 'onboarding' reason must be distinct from all prior reasons so these
    cases can be selected separately (AC)."""
    r = supervision.reasons_for("newcomer", len(SHORT), 200, [], False)
    assert r == ["onboarding"]
    assert "onboarding" not in ("budget", "short_answer", "no_memory_write",
                                "full", "control")


def test_onboarding_turns_end_to_end(db):
    """AC test: new tenant's first three turns in the DB; the fourth only if
    a signal fires."""
    supervision.capture("newbie", "ход 1", "ответ 1")
    supervision.capture("newbie", "ход 2", "ответ 2")
    supervision.capture("newbie", "ход 3", "ответ 3")
    supervision.capture("newbie", "ход 4", "ответ 4")  # no signal → not captured
    cases = supervision.pending(tenant="newbie")
    assert len(cases) == 3, "first three turns captured, fourth is not"
    assert all(c["reasons"] == "onboarding" for c in cases)
    assert [c["user"] for c in cases] == ["ход 1", "ход 2", "ход 3"]

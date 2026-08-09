"""Supervision capture — material for the offline critic (see dev ADR 0003).

Two capture modes: flagged tenants (SUPERVISED_TENANTS) get every turn stored;
everyone else only contributes turns that tripped a problem signal, plus a small
random control sample so the rubric doesn't drift toward only ever seeing bad cases.

This is the one place in the system that stores the *content* of a разбор, so:
content is Fernet-encrypted at rest with the same MEMORY_ENCRYPTION_KEY, the
tenant is the same hash used everywhere else, and rows expire after
SUPERVISION_RETENTION_DAYS.

Never raises into the coaching flow — a failed capture must not cost a user their answer.
"""

import asyncio
import json
import os
import random
import sqlite3
import time
from pathlib import Path

DB_FILE = Path(os.environ.get("SUPERVISION_DB", "supervision.db"))
RETENTION_DAYS = int(os.environ.get("SUPERVISION_RETENTION_DAYS", "90"))
SAMPLE_PCT = int(os.environ.get("SUPERVISION_SAMPLE_PCT", "5"))
SUPERVISED = {t.strip() for t in os.environ.get("SUPERVISED_TENANTS", "").split(",") if t.strip()}

# A разбор is "substantial" past this much user input — below it, a short answer
# and an empty memory are normal, not a signal.
SUBSTANTIAL_CHARS = 500
SHORT_ANSWER_CHARS = 600

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL NOT NULL,
    day         INTEGER NOT NULL,
    tenant      TEXT NOT NULL,       -- same hashed tenant id used everywhere
    skill       TEXT,                -- skill actually applied (via load_skill)
    reasons     TEXT NOT NULL,       -- why captured: budget,short_answer,no_memory_write,full,control
    user_len    INTEGER NOT NULL,
    answer_len  INTEGER NOT NULL,
    tools       TEXT,                -- tool names called this turn
    payload     BLOB NOT NULL,       -- Fernet(json{user, answer}) or plain json
    reviewed    INTEGER NOT NULL DEFAULT 0,
    verdict     TEXT                 -- critic's itemised findings
);
CREATE INDEX IF NOT EXISTS idx_cases_reviewed ON cases(reviewed);
CREATE INDEX IF NOT EXISTS idx_cases_day      ON cases(day);

CREATE TABLE IF NOT EXISTS passes (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ts             REAL NOT NULL,
    day            INTEGER NOT NULL,
    tenant         TEXT NOT NULL,
    tool_names     TEXT,                -- CSV: recall,save_memory,edit_memory
    write_count    INTEGER DEFAULT 0,   -- successful save/edit calls
    write_attempts INTEGER DEFAULT 0,   -- total save/edit attempts
    recall_count   INTEGER DEFAULT 0,
    crashed        INTEGER DEFAULT 0,   -- 0/1
    error          TEXT,                -- if crashed=1
    payload        BLOB                 -- Fernet(json{user, answer, summary}) or plain
);
CREATE INDEX IF NOT EXISTS idx_passes_tenant ON passes(tenant);
CREATE INDEX IF NOT EXISTS idx_passes_day    ON passes(day);
"""


def _fernet():
    key = os.environ.get("MEMORY_ENCRYPTION_KEY", "")
    if not key:
        return None
    from cryptography.fernet import Fernet
    return Fernet(key.encode() if isinstance(key, str) else key)


def _conn():
    c = sqlite3.connect(DB_FILE, timeout=5)
    c.execute("PRAGMA journal_mode=WAL")
    return c


def init():
    with _conn() as c:
        c.executescript(_SCHEMA)


def reasons_for(tenant, user_len, answer_len, tools_used, budget_exhausted):
    """Why (if at all) this turn is worth keeping. Empty list = don't store."""
    if tenant in SUPERVISED:
        return ["full"]

    out = []
    if budget_exhausted:
        out.append("budget")
    if user_len > SUBSTANTIAL_CHARS and answer_len < SHORT_ANSWER_CHARS:
        out.append("short_answer")
    # no_memory_write removed (aicoach-dev#11): after #10 moved memory writes
    # to the recollect pass, the coach never calls save_memory, so this signal
    # would fire on every turn. The signal now lives in pass_stats() instead.
    if not out and random.randint(1, 100) <= SAMPLE_PCT:
        out.append("control")  # keep the critic calibrated on normal sessions too
    return out


def capture(tenant, user_text, answer, *, skill=None, tools_used=(), budget_exhausted=False):
    """Store a turn if it qualifies. Fire-and-forget: never raises."""
    try:
        reasons = reasons_for(tenant, len(user_text), len(answer), tools_used, budget_exhausted)
        if not reasons:
            return
        blob = json.dumps({"user": user_text, "answer": answer}, ensure_ascii=False).encode()
        f = _fernet()
        if f:
            blob = f.encrypt(blob)
        now = time.time()
        with _conn() as c:
            c.execute(
                "INSERT INTO cases (ts, day, tenant, skill, reasons, user_len, answer_len, tools, payload) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (now, int(now // 86400), tenant, skill, ",".join(reasons),
                 len(user_text), len(answer), ",".join(tools_used), blob),
            )
    except Exception as exc:
        print("[supervision] capture error:", exc, flush=True)


# Keep strong refs — a bare create_task can be garbage-collected mid-flight.
_tasks: set[asyncio.Task] = set()


def capture_async(*args, **kwargs):
    """Capture off the event loop. Supervision must never sit between the model
    and the person waiting for their разбор — sqlite+Fernet is disk work, and this
    service runs a single uvicorn worker, so a sync call here would stall every
    concurrent session. Losing a case on shutdown is an acceptable trade."""
    task = asyncio.create_task(asyncio.to_thread(capture, *args, **kwargs))
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)


def capture_no_write(tenant, user_text, answer):
    """Queue a specific case when the recollect pass wrote nothing on
    substantial input (aicoach-dev#11 follow-up). pass_stats().no_write is
    an aggregate — it says "something's off this week" but not which turn.
    This puts the actual turn in front of the critic, same as the old
    turn-based no_memory_write signal did before #10 moved writes out of
    the turn. Below SUBSTANTIAL_CHARS an empty write is normal, not a signal."""
    if len(user_text) <= SUBSTANTIAL_CHARS:
        return
    try:
        blob = json.dumps({"user": user_text, "answer": answer}, ensure_ascii=False).encode()
        f = _fernet()
        if f:
            blob = f.encrypt(blob)
        now = time.time()
        with _conn() as c:
            c.execute(
                "INSERT INTO cases (ts, day, tenant, skill, reasons, user_len, answer_len, tools, payload) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (now, int(now // 86400), tenant, None, "no_memory_write",
                 len(user_text), len(answer), "", blob),
            )
    except Exception as exc:
        print("[supervision] capture_no_write error:", exc, flush=True)


def capture_no_write_async(*args, **kwargs):
    """Capture off the event loop. Like capture_async."""
    task = asyncio.create_task(asyncio.to_thread(capture_no_write, *args, **kwargs))
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)


def log_pass(tenant, *, tool_names=(), write_count=0, write_attempts=0,
             recall_count=0, crashed=False, error=None, user_text=None, answer=None):
    """Log a memory extraction pass. Fire-and-forget: never raises.

    Tracks what the recollect pass did after a coaching turn — which tools it
    called, how many writes succeeded, whether it crashed. Separate from cases
    (the coaching turn itself) so the no_memory_write signal can check pass
    results instead of turn tools (aicoach-dev#11).
    """
    try:
        summary = {
            "tool_names": list(tool_names),
            "write_count": write_count,
            "write_attempts": write_attempts,
            "recall_count": recall_count,
        }
        blob = json.dumps({"user": user_text, "answer": answer, "summary": summary},
                          ensure_ascii=False).encode()
        f = _fernet()
        if f:
            blob = f.encrypt(blob)
        now = time.time()
        with _conn() as c:
            c.execute(
                "INSERT INTO passes (ts, day, tenant, tool_names, write_count, "
                "write_attempts, recall_count, crashed, error, payload) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (now, int(now // 86400), tenant, ",".join(tool_names),
                 write_count, write_attempts, recall_count, int(crashed), error, blob),
            )
    except Exception as exc:
        print("[supervision] log_pass error:", exc, flush=True)


def log_pass_async(*args, **kwargs):
    """Log pass off the event loop. Like capture_async."""
    task = asyncio.create_task(asyncio.to_thread(log_pass, *args, **kwargs))
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)


def _decode(blob):
    f = _fernet()
    if f:
        blob = f.decrypt(blob)
    return json.loads(blob)


def pending(limit=50, tenant=None):
    """Un-reviewed cases, oldest first, with content decrypted. Optionally scoped
    to one tenant (supervise a single person's разборы rather than the whole pool)."""
    where = "reviewed = 0" + (" AND tenant = ?" if tenant else "")
    params = ([tenant, limit] if tenant else [limit])
    with _conn() as c:
        rows = c.execute(
            f"SELECT id, ts, tenant, skill, reasons, tools, payload FROM cases "
            f"WHERE {where} ORDER BY ts LIMIT ?", params
        ).fetchall()
    out = []
    for cid, ts, tenant, skill, reasons, tools_used, payload in rows:
        try:
            body = _decode(payload)
        except Exception as exc:
            print(f"[supervision] case {cid} unreadable:", exc, flush=True)
            continue
        out.append({"id": cid, "ts": ts, "tenant": tenant, "skill": skill,
                    "reasons": reasons, "tools": tools_used,
                    "user": body["user"], "answer": body["answer"]})
    return out


def mark_reviewed(case_id, verdict):
    with _conn() as c:
        c.execute("UPDATE cases SET reviewed = 1, verdict = ? WHERE id = ?", (verdict, case_id))


def purge(days=RETENTION_DAYS):
    """Drop cases past retention. Returns how many were removed."""
    cutoff = int(time.time() // 86400) - days
    with _conn() as c:
        cur = c.execute("DELETE FROM cases WHERE day < ?", (cutoff,))
        return cur.rowcount


def stats():
    with _conn() as c:
        total = c.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
        waiting = c.execute("SELECT COUNT(*) FROM cases WHERE reviewed = 0").fetchone()[0]
        by_reason = c.execute(
            "SELECT reasons, COUNT(*) FROM cases GROUP BY reasons ORDER BY 2 DESC"
        ).fetchall()
        # per-tenant breakdown drives the "which tenant, is there enough?" decision
        by_tenant = c.execute(
            "SELECT tenant, COUNT(*), SUM(CASE WHEN reviewed = 0 THEN 1 ELSE 0 END) "
            "FROM cases GROUP BY tenant ORDER BY 3 DESC"
        ).fetchall()
    return {"total": total, "pending": waiting, "by_reason": by_reason,
            "by_tenant": by_tenant}


def pass_stats(days=7):
    """Stats for memory extraction passes over the last N days.

    The no_memory_write signal now lives here (aicoach-dev#11): a pass that ran
    but wrote nothing (write_count=0, crashed=0) is the new signal, replacing
    the old turn-based check that broke after memory writes moved out of the
    coaching turn (#10).
    """
    cutoff = time.time() - days * 86400
    with _conn() as c:
        total = c.execute("SELECT COUNT(*) FROM passes WHERE ts > ?", (cutoff,)).fetchone()[0]
        crashed = c.execute("SELECT COUNT(*) FROM passes WHERE ts > ? AND crashed = 1",
                            (cutoff,)).fetchone()[0]
        no_write = c.execute(
            "SELECT COUNT(*) FROM passes WHERE ts > ? AND write_count = 0 AND crashed = 0",
            (cutoff,)
        ).fetchone()[0]
        by_tenant = c.execute(
            "SELECT tenant, COUNT(*), SUM(CASE WHEN write_count > 0 THEN 1 ELSE 0 END) "
            "FROM passes WHERE ts > ? GROUP BY tenant ORDER BY 2 DESC",
            (cutoff,)
        ).fetchall()
    return {"total": total, "crashed": crashed, "no_write": no_write, "by_tenant": by_tenant}

"""Privacy-safe engagement analytics for the AICOACH bot — one flat events
table in SQLite. No message content, no raw chat_id: everything is keyed by
the same hashed tenant_id used for memory. Answers a small set of marketing
questions (activation, retention, source attribution, feature usage) without
touching what anyone actually said.

Runs synchronously and cheaply (WAL, single-row inserts) — called from the
bot's asyncio loop via asyncio.to_thread, never blocks message handling.
"""

import os
import sqlite3
import time
from pathlib import Path

DB_FILE = Path(os.environ.get("ANALYTICS_DB", "analytics.db"))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      REAL NOT NULL,
    day     INTEGER NOT NULL,       -- days since epoch (UTC), for cheap grouping
    tenant  TEXT NOT NULL,          -- hashed tenant_id (or shared TENANT_ID in private mode)
    event   TEXT NOT NULL,          -- first_seen | message | answer | command | memory_write |
                                     -- weekly_on | weekly_off | weekly_push | rate_limited | blocked | deleted
    kind    TEXT,                   -- message: text|voice
    cmd     TEXT,                   -- command: /memory, /export, ...
    mtype   TEXT,                   -- memory_write: self|pattern|coach|decision|...
    source  TEXT                    -- first_seen: /start <param> deep-link, or "" for direct
);
CREATE INDEX IF NOT EXISTS idx_events_tenant ON events(tenant);
CREATE INDEX IF NOT EXISTS idx_events_day    ON events(day);
CREATE INDEX IF NOT EXISTS idx_events_event  ON events(event);
"""


def _conn():
    c = sqlite3.connect(DB_FILE, timeout=5)
    c.execute("PRAGMA journal_mode=WAL")
    return c


def init():
    with _conn() as c:
        c.executescript(_SCHEMA)


def log(event, tenant, *, kind=None, cmd=None, mtype=None, source=None):
    """Fire-and-forget event write. Never raises into the caller — analytics
    must not be able to break the coaching flow."""
    now = time.time()
    try:
        with _conn() as c:
            c.execute(
                "INSERT INTO events (ts, day, tenant, event, kind, cmd, mtype, source) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (now, int(now // 86400), tenant, event, kind, cmd, mtype, source),
            )
    except Exception as exc:
        print("[analytics] log error:", exc, flush=True)


def has_seen(tenant) -> bool:
    with _conn() as c:
        row = c.execute(
            "SELECT 1 FROM events WHERE tenant = ? AND event = 'first_seen' LIMIT 1", (tenant,)
        ).fetchone()
    return row is not None


# ---- queries for /stats and the CLI --------------------------------------

def _today():
    return int(time.time() // 86400)


def summary(days=30):
    """Headline numbers for the /stats command."""
    today = _today()
    since = today - days
    with _conn() as c:
        dau = c.execute(
            "SELECT COUNT(DISTINCT tenant) FROM events WHERE day = ? AND event='message'", (today,)
        ).fetchone()[0]
        wau = c.execute(
            "SELECT COUNT(DISTINCT tenant) FROM events WHERE day > ? AND event='message'", (today - 7,)
        ).fetchone()[0]
        mau = c.execute(
            "SELECT COUNT(DISTINCT tenant) FROM events WHERE day > ? AND event='message'", (since,)
        ).fetchone()[0]
        total_users = c.execute(
            "SELECT COUNT(DISTINCT tenant) FROM events WHERE event='first_seen'"
        ).fetchone()[0]

        new_7d = c.execute(
            "SELECT COUNT(*) FROM events WHERE day > ? AND event='first_seen'", (today - 7,)
        ).fetchone()[0]
        by_source = c.execute(
            "SELECT COALESCE(NULLIF(source,''),'direct'), COUNT(*) FROM events "
            "WHERE day > ? AND event='first_seen' GROUP BY 1 ORDER BY 2 DESC", (today - 7,)
        ).fetchall()

        # activation: of users first_seen in the window, share with >=1 message
        new_users = [r[0] for r in c.execute(
            "SELECT tenant FROM events WHERE day > ? AND event='first_seen'", (since,)
        ).fetchall()]
        activated = 0
        for t in new_users:
            if c.execute("SELECT 1 FROM events WHERE tenant=? AND event='message' LIMIT 1", (t,)).fetchone():
                activated += 1
        activation_pct = round(100 * activated / len(new_users)) if new_users else None

        # retention: of users first_seen on day D, share with a message on D+1 / D+7
        def retention(offset):
            cohort_days = range(today - days, today - offset)
            hit, total = 0, 0
            for d in cohort_days:
                cohort = [r[0] for r in c.execute(
                    "SELECT tenant FROM events WHERE day=? AND event='first_seen'", (d,)
                ).fetchall()]
                total += len(cohort)
                for t in cohort:
                    if c.execute(
                        "SELECT 1 FROM events WHERE tenant=? AND event='message' AND day=? LIMIT 1",
                        (t, d + offset),
                    ).fetchone():
                        hit += 1
            return round(100 * hit / total) if total else None

        d1 = retention(1)
        d7 = retention(7)

        turns = c.execute(
            "SELECT COUNT(*) FROM events WHERE day > ? AND event='message'", (since,)
        ).fetchone()[0]
        voice = c.execute(
            "SELECT COUNT(*) FROM events WHERE day > ? AND event='message' AND kind='voice'", (since,)
        ).fetchone()[0]
        active_users_count = mau or 1
        depth = round(turns / active_users_count, 1)
        voice_pct = round(100 * voice / turns) if turns else None

        mem_users = c.execute(
            "SELECT COUNT(DISTINCT tenant) FROM events WHERE day > ? AND event='memory_write'", (since,)
        ).fetchone()[0]
        mem_pct = round(100 * mem_users / active_users_count) if active_users_count else None

        weekly_on = c.execute("""
            SELECT COUNT(DISTINCT tenant) FROM events e1 WHERE event='weekly_on'
              AND NOT EXISTS (
                SELECT 1 FROM events e2 WHERE e2.tenant=e1.tenant AND e2.event='weekly_off'
                  AND e2.ts > e1.ts
              )
        """).fetchone()[0]
        pushes = c.execute(
            "SELECT COUNT(*) FROM events WHERE day > ? AND event='weekly_push'", (since,)
        ).fetchone()[0]
        pushed_tenants = [r[0] for r in c.execute(
            "SELECT DISTINCT tenant FROM events WHERE day > ? AND event='weekly_push'", (since,)
        ).fetchall()]
        returned = 0
        for t in pushed_tenants:
            last_push = c.execute(
                "SELECT MAX(ts) FROM events WHERE tenant=? AND event='weekly_push'", (t,)
            ).fetchone()[0]
            if c.execute(
                "SELECT 1 FROM events WHERE tenant=? AND event='message' AND ts>? LIMIT 1",
                (t, last_push),
            ).fetchone():
                returned += 1
        push_return_pct = round(100 * returned / len(pushed_tenants)) if pushed_tenants else None

        cmd_counts = c.execute(
            "SELECT cmd, COUNT(*) FROM events WHERE day > ? AND event='command' AND cmd IS NOT NULL "
            "GROUP BY cmd ORDER BY 2 DESC", (since,)
        ).fetchall()

    return {
        "days": days, "dau": dau, "wau": wau, "mau": mau, "total_users": total_users,
        "new_7d": new_7d, "by_source": by_source,
        "activation_pct": activation_pct, "d1_retention": d1, "d7_retention": d7,
        "turns_per_active": depth, "voice_pct": voice_pct, "memory_write_pct": mem_pct,
        "weekly_subs": weekly_on, "weekly_pushes": pushes, "weekly_return_pct": push_return_pct,
        "commands": cmd_counts,
    }


def format_summary(s) -> str:
    src = " · ".join(f"{name} {n}" for name, n in s["by_source"]) or "—"
    cmds = " ".join(f"{c} {n}" for c, n in s["commands"]) or "—"
    def pct(x):
        return f"{x}%" if x is not None else "—"
    return (
        f"📊 AICOACH · последние {s['days']} дней\n\n"
        f"Юзеры: DAU {s['dau']} · WAU {s['wau']} · MAU {s['mau']} · всего {s['total_users']}\n"
        f"Новые за 7д: {s['new_7d']} → {src}\n\n"
        f"Активация (≥1 разбор): {pct(s['activation_pct'])}\n"
        f"Retention: D1 {pct(s['d1_retention'])} · D7 {pct(s['d7_retention'])}\n\n"
        f"Глубина: {s['turns_per_active']} обменов/актив · голос {pct(s['voice_pct'])}\n"
        f"Память выросла у {pct(s['memory_write_pct'])} активных\n\n"
        f"Weekly: {s['weekly_subs']} подписаны · возврат по пушу {pct(s['weekly_return_pct'])}\n"
        f"Команды: {cmds}"
    )

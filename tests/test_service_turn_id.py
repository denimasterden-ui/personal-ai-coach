"""T — turn_id flows from request through brain to quality layer (aicoach-dev#13).

Seam: service._run_inner with a substituted model client.
Assertions: the identifier from the request lands in the ``answer`` SSE event
and in the captured supervision case — without touching the real LLM.
"""
import json
from types import SimpleNamespace

import pytest

import service
import supervision


# ── fakes ────────────────────────────────────────────────────────────────────

class FakeMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self, exclude_none=False):
        d = {"role": "assistant", "content": self.content}
        if exclude_none:
            d = {k: v for k, v in d.items() if v is not None}
        return d


class FakeClient:
    def __init__(self, replies):
        self._replies = list(replies)
        self._idx = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kw):
        msg = self._replies[self._idx]
        self._idx += 1
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse(chunk):
    if not chunk.startswith("event: "):
        return None, None
    nl = chunk.index("\n")
    event = chunk[len("event: "):nl]
    rest = chunk[nl + 1:]
    data = json.loads(rest[len("data: "):].rstrip("\n")) if rest.startswith("data: ") else {}
    return event, data


async def _play(req):
    events = {}
    async for chunk in service._run_inner(req):
        event, data = _parse(chunk)
        if event:
            events[event] = data
    return events


# ── fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Real supervision DB in tmp_path; fake tool dispatch; real capture path."""
    import config
    import tools

    monkeypatch.setattr(config, "TENANTS_DIR", tmp_path / "tenants")
    (tmp_path / "tenants").mkdir()
    monkeypatch.setattr(tools, "dispatch", lambda name, args, tenant: {"ok": True})
    monkeypatch.setattr(supervision, "DB_FILE", tmp_path / "sup.db")
    monkeypatch.setattr(supervision, "SUPERVISED", {"t1"})  # capture every turn
    supervision.init()


def _use_client(monkeypatch, replies):
    client = FakeClient(replies)
    monkeypatch.setattr(service, "_client", client)
    return client


# ── tests ─────────────────────────────────────────────────────────────────────

async def test_turn_id_appears_in_answer_event(isolated, monkeypatch):
    """Brain echoes the bot's turn_id in the SSE answer event."""
    _use_client(monkeypatch, [FakeMessage(content="ответ коуча")])
    req = service.SessionRequest(tenant_id="t1", message="привет", turn_id="abc123")
    events = await _play(req)
    assert events["answer"]["turn_id"] == "abc123"


async def test_turn_id_reaches_supervision_case(isolated, monkeypatch):
    """turn_id from the request lands in the supervision DB case row."""
    _use_client(monkeypatch, [FakeMessage(content="ответ коуча")])
    req = service.SessionRequest(tenant_id="t1", message="привет", turn_id="abc123")
    await _play(req)
    # capture_async is async; wait for the task
    import asyncio
    await asyncio.sleep(0.05)
    cases = supervision.pending(tenant="t1")
    assert cases, "expected at least one case for supervised tenant"
    assert cases[0]["turn_id"] == "abc123"


async def test_turn_id_in_budget_exhausted_path(isolated, monkeypatch):
    """Budget-exhausted fallback must not lose the turn_id."""
    from types import SimpleNamespace as NS

    def _call(name):
        return NS(id=f"call-{name}",
                  function=NS(name=name, arguments="{}"))

    class FakeMsgWithTools(FakeMessage):
        def model_dump(self, exclude_none=False):
            d = super().model_dump(exclude_none=exclude_none)
            d["tool_calls"] = [{"id": c.id, "type": "function",
                                 "function": {"name": c.function.name,
                                              "arguments": c.function.arguments}}
                                for c in (self.tool_calls or [])]
            if exclude_none:
                d = {k: v for k, v in d.items() if v is not None}
            return d

    script = [FakeMsgWithTools(content=f"шаг {i}", tool_calls=[_call("recall")])
              for i in range(service.MAX_STEPS)]
    script.append(FakeMessage(content="финальный хвост"))
    _use_client(monkeypatch, script)

    req = service.SessionRequest(tenant_id="t1", message="a" * 600, turn_id="budget-turn-42")
    events = await _play(req)
    assert events["answer"]["turn_id"] == "budget-turn-42"
    import asyncio
    await asyncio.sleep(0.05)
    cases = supervision.pending(tenant="t1")
    assert cases, "budget-exhausted turn must be captured"
    assert cases[0]["turn_id"] == "budget-turn-42"


async def test_turn_id_auto_generated_if_absent(isolated, monkeypatch):
    """If the request carries no turn_id, the brain generates one."""
    _use_client(monkeypatch, [FakeMessage(content="ответ")])
    req = service.SessionRequest(tenant_id="t1", message="без id")
    events = await _play(req)
    tid = events["answer"].get("turn_id")
    assert tid and len(tid) == 32  # uuid4().hex


async def test_existing_case_without_turn_id_is_readable(isolated, monkeypatch):
    """Rows inserted before the migration (turn_id=NULL) must still be readable."""
    import time
    now = time.time()
    with supervision._conn() as c:
        c.execute(
            "INSERT INTO cases (ts, day, tenant, reasons, user_len, answer_len, payload) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (now, int(now // 86400), "t1", "full", 10, 10, b'{"user":"q","answer":"a"}'),
        )
    cases = supervision.pending(tenant="t1")
    assert len(cases) == 1
    assert cases[0]["user"] == "q"
    assert cases[0]["turn_id"] is None  # absent before migration — not an error

"""B1 — разбор доходит при исчерпании бюджета шагов (aicoach-dev#9).

External seam only: a full turn through ``service._run_inner`` with a substituted
client. Assertions touch only what reached the human (the ``answer`` event) and
what the quality layer saw (``budget_exhausted``) — never prompt text or call
counts. ``bot.py`` reads only the ``answer`` event, so whatever the human must
see has to land in that one event.
"""
import json
from types import SimpleNamespace

import pytest

import service


# ── fakes ────────────────────────────────────────────────────────────────────

class FakeMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls
        self.role = "assistant"

    def model_dump(self, exclude_none=False):
        d = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [{
                "id": c.id, "type": "function",
                "function": {"name": c.function.name, "arguments": c.function.arguments},
            } for c in self.tool_calls]
        if exclude_none:
            d = {k: v for k, v in d.items() if v is not None}
        return d


def _call(name):
    return SimpleNamespace(
        id=f"call-{name}", function=SimpleNamespace(name=name, arguments="{}"))


class FakeClient:
    def __init__(self, script):
        self.script = script
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(choices=[SimpleNamespace(
            message=self.script[len(self.calls) - 1])])


# ── seam: drive _run_inner and collect what reached the human ────────────────

def _parse(chunk):
    if not chunk.startswith("event: "):
        return None, None
    nl = chunk.index("\n")
    event = chunk[len("event: "):nl]
    rest = chunk[nl + 1:]
    data = json.loads(rest[len("data: "):].rstrip("\n")) if rest.startswith("data: ") else {}
    return event, data


async def _play(req):
    seen = {"thoughts": [], "answer": None, "tool_calls": []}
    async for chunk in service._run_inner(req):
        event, data = _parse(chunk)
        if event == "thought":
            seen["thoughts"].append(data["text"])
        elif event == "answer":
            seen["answer"] = data["text"]
        elif event == "tool_call":
            seen["tool_calls"].append(data["name"])
    return seen


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Steer every collaborator _run_inner touches to a fake: tool dispatch (no
    real memory write), supervision capture (no db write), tenant dir (nothing
    escapes tmp). The model client is set per test via _use_client."""
    import config
    import supervision
    import tools

    monkeypatch.setattr(config, "TENANTS_DIR", tmp_path / "tenants")
    (tmp_path / "tenants").mkdir()
    monkeypatch.setattr(tools, "dispatch", lambda name, args, tenant: {"ok": True})
    captured = []

    def fake_capture(*a, **k):
        captured.append({"args": a, "kwargs": k})

    monkeypatch.setattr(supervision, "capture_async", fake_capture)
    return captured


def _use_client(monkeypatch, script):
    client = FakeClient(script)
    monkeypatch.setattr(service, "_client", client)
    return client


# ── tests ────────────────────────────────────────────────────────────────────

async def test_budget_exhausted_delivers_accumulated_draft(isolated, monkeypatch):
    """The разбор drafted across tool steps must reach the human when the step
    budget runs out — not only the final реплика."""
    long_chunk = "часть разбора. " * 60  # > 800 chars: the old gate would drop this
    script = []
    for i in range(service.MAX_STEPS):
        script.append(FakeMessage(content=(long_chunk if i == 0 else f"шаг разбора {i}"),
                                  tool_calls=[_call("save_memory")]))
    script.append(FakeMessage(content="завершающий разбор человеку"))
    _use_client(monkeypatch, script)

    req = service.SessionRequest(tenant_id="t1", message="длинное голосовое")
    seen = await _play(req)

    assert long_chunk in seen["answer"], ">800-char intermediate text must reach the human"
    assert "шаг разбора 1" in seen["answer"]
    assert "завершающий разбор" in seen["answer"]
    assert isolated[-1]["kwargs"].get("budget_exhausted") is True


async def test_normal_completion_unchanged(isolated, monkeypatch):
    """A turn that finishes inside the budget answers exactly as before."""
    _use_client(monkeypatch, [FakeMessage(content="короткий ответ")])
    req = service.SessionRequest(tenant_id="t1", message="привет")
    seen = await _play(req)
    assert seen["answer"] == "короткий ответ"
    assert seen["thoughts"] == []
    assert not isolated[-1]["kwargs"].get("budget_exhausted")


async def test_normal_turn_with_tools_keeps_draft_out_of_answer(isolated, monkeypatch):
    """The fix is scoped to budget exhaustion: a normal tool-then-answer turn must
    not leak the intermediate draft into the answer."""
    _use_client(monkeypatch, [
        FakeMessage(content="промежуточный набросок", tool_calls=[_call("recall")]),
        FakeMessage(content="финальный разбор"),
    ])
    req = service.SessionRequest(tenant_id="t1", message="расскажи про мой день")
    seen = await _play(req)
    assert seen["answer"] == "финальный разбор"
    assert "промежуточный набросок" not in seen["answer"]

"""Memory extraction pass (aicoach-dev#10).

External seam: a full turn through ``service._run_inner`` with a substituted
client. Assertions touch only what reached the human (the ``answer`` event),
what tools the coach saw (``TURN_TOOLS`` excludes write tools), and what
landed on disk after the pass completed. Never peek inside prompts or count
model calls.

The fake client scripts the model's responses; no network, no real key.
"""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import service
import tools


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


def _call(name, args, cid=None):
    return SimpleNamespace(
        id=cid or f"call-{name}",
        function=SimpleNamespace(name=name, arguments=json.dumps(args, ensure_ascii=False)),
    )


class FakeClient:
    """`script` is either a list of FakeMessage (consumed in call order) or a
    callable(create_kwargs) -> FakeMessage. Records every create call."""

    def __init__(self, script):
        self.script = script
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kw):
        self.calls.append(kw)
        if callable(self.script):
            msg = self.script(kw)
        else:
            msg = self.script[len(self.calls) - 1]
        if isinstance(msg, BaseException):
            raise msg
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


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

    monkeypatch.setattr(config, "TENANTS_DIR", tmp_path / "tenants")
    (tmp_path / "tenants").mkdir()
    captured = []

    def fake_capture(*a, **k):
        captured.append({"args": a, "kwargs": k})

    monkeypatch.setattr(supervision, "capture_async", fake_capture)
    return captured


def _use_client(monkeypatch, script):
    client = FakeClient(script)
    monkeypatch.setattr(service, "_client", client)
    return client


# ── external seam tests ──────────────────────────────────────────────────────

async def test_turn_has_no_write_tools(isolated, monkeypatch):
    """The coach cannot write memory during the turn — the tool schemas it sees
    do not include save_memory or edit_memory. Observable effect: the model's
    `tools` kwarg excludes write tools (AC1)."""
    client = _use_client(monkeypatch, [FakeMessage(content="разбор ситуации")])
    req = service.SessionRequest(tenant_id="t1", message="расскажи про мой день")
    await _play(req)

    # Model saw only read tools (no write tools)
    call_kw = client.calls[0]
    tool_names = {t["function"]["name"] for t in call_kw["tools"]}
    assert "save_memory" not in tool_names, "coach has no save_memory tool"
    assert "edit_memory" not in tool_names, "coach has no edit_memory tool"
    assert "recall" in tool_names, "coach can read memory"


async def test_answer_does_not_start_with_write_report(isolated, monkeypatch):
    """The answer must not open with a report about what was written. The coach
    has no write tools, so it can't write and can't report writing."""
    long_answer = "вот подробный разбор твоей ситуации. " * 3
    _use_client(monkeypatch, [FakeMessage(content=long_answer)])
    req = service.SessionRequest(tenant_id="t1", message="что происходит")
    seen = await _play(req)

    assert seen["answer"] is not None
    assert not seen["answer"].lower().startswith("записал")
    assert "записал в" not in seen["answer"].lower()
    assert len(seen["answer"]) > 50, "substantive answer, not a write report"


async def test_memory_written_after_turn(isolated, monkeypatch, tmp_path):
    """After the turn completes, the recollect pass writes memory to disk.
    The pass runs in background; we wait for it to finish."""
    import recollect

    # Isolate: prevent _run_inner from firing the pass (we'll fire it manually
    # with a properly scripted client). Save the original so we can call it.
    orig_after_turn = recollect.after_turn_async
    monkeypatch.setattr(recollect, "after_turn_async", lambda *a, **k: None)

    # Coach gives a substantive answer
    _use_client(monkeypatch, [FakeMessage(content="разбор: ты упомянул важный паттерн...")])

    req = service.SessionRequest(tenant_id="t1", message="у меня повторяется ситуация")
    await _play(req)

    # Pass writes memory (scripted to call save_memory)
    pass_client = FakeClient([
        FakeMessage(content="проверяю", tool_calls=[_call("recall", {"query": "паттерн"})]),
        FakeMessage(content="записываю", tool_calls=[_call("save_memory", {
            "type": "pattern", "slug": "important-pattern",
            "content": "Важный паттерн из разговора",
            "source": "слова человека в сессии"})]),
        FakeMessage(content="готово"),
    ])

    # Manually trigger the pass with the scripted client (using the original)
    orig_after_turn(
        "t1", "у меня повторяется ситуация", "разбор: ты упомянул важный паттерн...",
        client=pass_client, model="test-model"
    )
    # Wait for background tasks
    await asyncio.sleep(0.1)

    # Verify memory was written
    import config
    pattern_file = config.TENANTS_DIR / "t1" / "patterns" / "important-pattern.md"
    assert pattern_file.exists(), "pass wrote memory to disk"
    content = pattern_file.read_text(encoding="utf-8")
    assert "Важный паттерн" in content


async def test_pass_crash_does_not_affect_answer(isolated, monkeypatch):
    """A crashed recollect pass must not affect the answer that was already
    delivered. The turn is complete before the pass runs."""
    import recollect

    # Isolate: prevent _run_inner from firing the pass (we'll fire a crashing
    # one manually). Save the original so we can call it.
    orig_after_turn = recollect.after_turn_async
    monkeypatch.setattr(recollect, "after_turn_async", lambda *a, **k: None)

    _use_client(monkeypatch, [FakeMessage(content="полный разбор ситуации")])

    req = service.SessionRequest(tenant_id="t1", message="что происходит")
    seen = await _play(req)

    # Answer was delivered
    assert seen["answer"] == "полный разбор ситуации"

    # Manually fire a crashing pass — the answer is already gone
    crashing_client = FakeClient([RuntimeError("pass crashed")])
    orig_after_turn(
        "t1", "что происходит", seen["answer"],
        client=crashing_client, model="test-model"
    )
    await asyncio.sleep(0.1)

    # Answer unchanged — pass failure is logged, not raised
    assert seen["answer"] == "полный разбор ситуации"


# ── internal seam tests ────────────────────────────────────────────────────
# These drive ``recollect._run_pass_inner`` directly (the pass's actual loop,
# not the fire-and-forget wrapper) and verify what landed on disk. No peeking
# inside prompts or counting model calls — only observable effect.

async def test_pass_retries_after_gate_rejection(isolated):
    """First save_memory is missing source → gate rejects → pass retries with
    source → memory lands on disk. The pass has a 3-step budget exactly so
    one refusal doesn't cost a write."""
    import recollect
    import config

    client = FakeClient([
        # Step 1: save without source — gate refuses
        FakeMessage(content="записываю", tool_calls=[_call("save_memory", {
            "type": "pattern", "slug": "retry-test",
            "content": "важный наблюдение",
            # missing source
        }, cid="call-1")]),
        # Step 2: retry with source — gate accepts
        FakeMessage(content="повторяю", tool_calls=[_call("save_memory", {
            "type": "pattern", "slug": "retry-test",
            "content": "важное наблюдение",
            "source": "слова человека в сессии",
        }, cid="call-2")]),
        # Step 3: done
        FakeMessage(content="готово"),
    ])

    await recollect._run_pass_inner(
        "t1", "у меня есть наблюдение", "разбор",
        client=client, model="test-model"
    )

    pattern_file = config.TENANTS_DIR / "t1" / "patterns" / "retry-test.md"
    assert pattern_file.exists(), "retry after gate rejection succeeded"
    content = pattern_file.read_text(encoding="utf-8")
    assert "важное наблюдение" in content


async def test_pass_dedups_via_recall(isolated):
    """Pre-populated memory + pass that decides not to write → no duplicate
    on disk. The pass's dedup is exercised through recall returning existing
    content, and the model choosing to stop."""
    import recollect
    import config

    # Pre-populate: this pattern already exists
    tenant_dir = config.TENANTS_DIR / "t1" / "patterns"
    tenant_dir.mkdir(parents=True)
    (tenant_dir / "existing.md").write_text(
        "существующий паттерн\n\n_source: слова человека в сессии_\n",
        encoding="utf-8"
    )

    client = FakeClient([
        # Step 1: recall for dedup — returns existing content
        FakeMessage(content="проверяю", tool_calls=[_call("recall", {"query": "паттерн"})]),
        # Step 2: pass decides nothing new to write (no tool calls)
        FakeMessage(content="уже есть, не пишу"),
    ])

    await recollect._run_pass_inner(
        "t1", "у меня тот же паттерн", "разбор",
        client=client, model="test-model"
    )

    # Only the original file — no duplicate
    patterns = list(tenant_dir.glob("*.md"))
    assert len(patterns) == 1, "no duplicate written"
    assert patterns[0].name == "existing.md"


async def test_explicit_write_request(isolated):
    """A turn where the human explicitly said «запиши это» must produce a write.
    The rule lives in the pass prompt, not in a separate interface."""
    import recollect
    import config

    client = FakeClient([
        # Step 1: pass recalls (dedup check — nothing found)
        FakeMessage(content="проверяю", tool_calls=[_call("recall", {"query": "решение"})]),
        # Step 2: pass writes the explicit request
        FakeMessage(content="записываю", tool_calls=[_call("save_memory", {
            "type": "decision", "slug": "fire-contractor",
            "content": "Решение: уволить подрядчика",
            "source": "слова человека в сессии",
        })]),
        # Step 3: done
        FakeMessage(content="готово"),
    ])

    await recollect._run_pass_inner(
        "t1",
        "Запиши это: я решил уволить подрядчика",
        "Понял, это серьёзное решение. Что привело тебя к нему?",
        client=client, model="test-model"
    )

    decision_file = config.TENANTS_DIR / "t1" / "decisions" / "fire-contractor.md"
    assert decision_file.exists(), "explicit write request produced a write"
    content = decision_file.read_text(encoding="utf-8")
    assert "уволить подрядчика" in content


# ── passes trace (aicoach-dev#11) ────────────────────────────────────────────
# These verify that the recollect pass logs its activity to supervision.passes.


async def test_pass_summary_reports_writes(isolated):
    """_run_pass_inner returns a summary with tool_names, write_count, etc."""
    import recollect

    client = FakeClient([
        # Step 1: recall
        FakeMessage(content="проверяю", tool_calls=[_call("recall", {"query": "паттерн"})]),
        # Step 2: save_memory (success)
        FakeMessage(content="записываю", tool_calls=[_call("save_memory", {
            "type": "pattern", "slug": "test-pattern",
            "content": "тестовый паттерн",
            "source": "слова человека в сессии",
        })]),
        # Step 3: done
        FakeMessage(content="готово"),
    ])

    summary = await recollect._run_pass_inner(
        "t1", "сообщение", "ответ",
        client=client, model="test-model"
    )

    assert summary["tool_names"] == ["recall", "save_memory"]
    assert summary["write_count"] == 1
    assert summary["write_attempts"] == 1
    assert summary["recall_count"] == 1


async def test_pass_logs_to_supervision(isolated, monkeypatch):
    """After the pass completes, supervision.passes contains a record."""
    import recollect
    import supervision

    # Mock log_pass_async to capture calls
    logged = []

    def mock_log_pass_async(*args, **kwargs):
        logged.append({"args": args, "kwargs": kwargs})

    monkeypatch.setattr(supervision, "log_pass_async", mock_log_pass_async)

    client = FakeClient([
        FakeMessage(content="записываю", tool_calls=[_call("save_memory", {
            "type": "pattern", "slug": "logged-pattern",
            "content": "паттерн для лога",
            "source": "слова человека в сессии",
        })]),
        FakeMessage(content="готово"),
    ])

    await recollect._run_pass("t1", "сообщение", "ответ", client=client, model="test-model")

    assert len(logged) == 1
    assert logged[0]["args"][0] == "t1"  # tenant_id is positional
    assert logged[0]["kwargs"]["write_count"] == 1
    assert logged[0]["kwargs"]["crashed"] is False


async def test_pass_crash_logs_to_supervision(isolated, monkeypatch):
    """A crashed pass logs crashed=1 with error message."""
    import recollect
    import supervision

    logged = []

    def mock_log_pass_async(*args, **kwargs):
        logged.append({"args": args, "kwargs": kwargs})

    monkeypatch.setattr(supervision, "log_pass_async", mock_log_pass_async)

    # Client that raises on first call
    client = FakeClient([RuntimeError("pass crashed mid-flight")])

    await recollect._run_pass("t1", "сообщение", "ответ", client=client, model="test-model")

    assert len(logged) == 1
    assert logged[0]["kwargs"]["crashed"] is True
    assert "pass crashed mid-flight" in logged[0]["kwargs"]["error"]
    assert logged[0]["kwargs"]["write_count"] == 0

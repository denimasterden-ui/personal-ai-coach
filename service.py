"""AICOACH: personal coaching brain. One endpoint. A bare Claude with a coaching
skill and hands to its own memory (recall / save_memory / load_skill) — the model
runs the conversation freely and decides when to remember and when to write. No
deterministic pipe, no data/sandbox tools.

Trimmed fork of ops-agent/service.py: kept the SSE stream, prompt caching, the
tool-loop and per-turn context compaction; dropped run_python/sandbox/discover,
the Neo4j session persistence, and the dashboard/browser-link endpoints.
"""

import asyncio
import json
import os
import pickle
import time
from pathlib import Path

import tiktoken
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from openai import AsyncOpenAI
from pydantic import BaseModel

import config
import memory
import supervision
import tools

app = FastAPI()

_client = AsyncOpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY)
_enc = tiktoken.get_encoding("cl100k_base")

MAX_STEPS = 6            # coaching needs a few memory tool-calls, not 24 data steps
MAX_OUTPUT_TOKENS = 16000  # reasoning models share this budget with hidden reasoning
SESSION_TTL_SECONDS = 6 * 3600
PLAYBOOK_TITLE = "Integrative Personal Coaching v1"

# OpenRouter: pin throughput so we don't scatter across slow providers.
_EXTRA_BODY = {"provider": {"sort": "throughput"}}
if config.REASONING_EFFORT:
    _EXTRA_BODY["reasoning"] = {"effort": config.REASONING_EFFORT}


class SessionRequest(BaseModel):
    tenant_id: str
    message: str
    session_id: str | None = None
    model: str | None = None  # override CORE model (research-track R2 benchmark)


class _Session:
    __slots__ = ("messages", "last_used")

    def __init__(self, messages):
        self.messages = messages
        self.last_used = time.time()


_sessions: dict[str, _Session] = {}
_active = 0  # in-flight /session turns (model requests) — deploy gate reads this


def _evict_stale():
    now = time.time()
    for sid in [s for s, v in _sessions.items() if now - v.last_used > SESSION_TTL_SECONDS]:
        del _sessions[sid]


# Persist live conversations across restarts so a deploy doesn't wipe the
# in-RAM context of anyone mid-разбор. Encrypted at rest with the same key as
# memory when running the public demo (the dump holds conversation text).
SESSIONS_FILE = Path(os.environ.get("SESSIONS_FILE", "sessions.pkl"))
_sess_fernet = None
_enc_key = os.environ.get("MEMORY_ENCRYPTION_KEY")
if _enc_key:
    from cryptography.fernet import Fernet
    _sess_fernet = Fernet(_enc_key.encode())


def _persist_sessions():
    _evict_stale()
    try:
        data = pickle.dumps(_sessions)
        if _sess_fernet:
            data = _sess_fernet.encrypt(data)
        SESSIONS_FILE.write_bytes(data)
        print(f"[sessions] persisted {len(_sessions)}", flush=True)
    except Exception as exc:
        print("[sessions] persist error:", exc, flush=True)


def _restore_sessions():
    if not SESSIONS_FILE.exists():
        return
    try:
        data = SESSIONS_FILE.read_bytes()
        if _sess_fernet:
            data = _sess_fernet.decrypt(data)
        _sessions.update(pickle.loads(data))
        _evict_stale()
        print(f"[sessions] restored {len(_sessions)}", flush=True)
    except Exception as exc:
        print("[sessions] restore error:", exc, flush=True)


@app.on_event("startup")
async def _on_startup():
    _restore_sessions()
    supervision.init()
    dropped = supervision.purge()
    if dropped:
        print(f"[supervision] purged {dropped} case(s) past retention", flush=True)


@app.on_event("shutdown")
async def _on_shutdown():
    _persist_sessions()


def _sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _cache_block(text):
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


def _with_cache_control(msgs):
    """Mark system prompt + current last message as cache breakpoints (Anthropic
    prompt caching via OpenRouter). Rebuilt each call, never mutates history."""
    out = list(msgs)
    if isinstance(out[0].get("content"), str):
        out[0] = {**out[0], "content": _cache_block(out[0]["content"])}
    if len(out) > 1 and isinstance(out[-1].get("content"), str):
        out[-1] = {**out[-1], "content": _cache_block(out[-1]["content"])}
    return out


async def _build_system_prompt():
    playbook = await memory.load_skill(PLAYBOOK_TITLE) or ""
    catalog = await memory.load_skills_catalog()
    extra = [s for s in catalog if s["title"].strip().lower() != PLAYBOOK_TITLE.lower()]
    catalog_doc = (
        "\n".join(f"- **{s['title']}**: {s['description']}" for s in extra)
        if extra else "_Других скиллов пока нет._"
    )
    return (
        playbook
        + "\n\n---\n\n# Память пользователя\n"
        "У тебя есть инструменты к личной памяти этого пользователя. Зови `recall` в начале "
        "разбора и когда нужен контекст о человеке; записывай важные выводы/паттерны/решения через "
        "`save_memory`, чтобы профиль рос между сессиями. Для self/open_loops/evidence используй "
        "mode='append', чтобы дополнять (не затирать уже записанное — особенно если пользователь "
        "присылает профиль частями), и mode='replace' только когда факт устарел и его надо переписать. "
        "Указывай `supersedes`, если уточняешь старую запись.\n\n"
        "# Дополнительные скиллы (вызови load_skill(title) за полным текстом)\n\n"
        + catalog_doc
    )


async def _run(req: SessionRequest):
    """Thin wrapper: count the turn as an in-flight model request so a deploy
    can wait for a quiet window (active == 0) before restarting."""
    global _active
    _active += 1
    try:
        async for chunk in _run_inner(req):
            yield chunk
    finally:
        _active -= 1


async def _run_inner(req: SessionRequest):
    model_id = req.model or config.LLM_MODEL
    _evict_stale()
    session = _sessions.get(req.session_id) if req.session_id else None

    if session is None:
        system_prompt = await _build_system_prompt()
        session = _Session([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": req.message},
        ])
        if req.session_id:
            _sessions[req.session_id] = session
    else:
        session.messages.append({"role": "user", "content": req.message})
    session.last_used = time.time()
    messages = session.messages

    # Supervision context for this turn (ADR 0003): which skill actually fired and
    # which tools were called — the critic judges against the applied skill's own
    # "критерий качественного разбора", so it needs to know which one that was.
    used_tools: list[str] = []
    skill_used = None

    for _ in range(MAX_STEPS):
        response = await _client.chat.completions.create(
            model=model_id,
            messages=_with_cache_control(messages),
            tools=tools.TOOL_SCHEMAS,
            max_tokens=MAX_OUTPUT_TOKENS,
            extra_body=_EXTRA_BODY,
        )
        message = response.choices[0].message

        if message.tool_calls and message.content and len(message.content.strip()) < 800:
            yield _sse("thought", {"text": message.content})

        if not message.tool_calls:
            messages.append(message.model_dump(exclude_none=True))
            answer = message.content or ""
            supervision.capture_async(req.tenant_id, req.message, answer,
                                      skill=skill_used, tools_used=used_tools)
            yield _sse("answer", {"text": answer})
            return

        messages.append(message.model_dump(exclude_none=True))

        async def _exec(call):
            name = call.function.name
            try:
                args = json.loads(call.function.arguments or "{}")
                result = await tools.dispatch(name, args, req.tenant_id)
            except Exception as exc:
                args, result = {}, {"error": f"{type(exc).__name__}: {exc}"}
            return call, name, args, result

        for call, name, args, result in await asyncio.gather(
            *(_exec(c) for c in message.tool_calls)
        ):
            used_tools.append(name)
            if name == "load_skill":
                skill_used = args.get("title") or skill_used
            yield _sse("tool_call", {"name": name, "args": args})
            yield _sse("tool_result", {"name": name, "ok": not (isinstance(result, dict) and result.get("error"))})
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result, ensure_ascii=False, default=str),
            })

    # Step budget spent while the model still wanted to call tools — a big/rich
    # message (long voice transcript, multi-topic dump) can burn the budget on
    # recall+save_memory before it gets to an actual reply. Rather than forcing
    # a confident-sounding wrap-up (which reads as a rushed non-answer), tell
    # the model to be honest about running out of room — and log it so we can
    # see from real traffic (not a guess) whether MAX_STEPS needs raising.
    print(f"[budget] MAX_STEPS={MAX_STEPS} exhausted mid-tool-calls, tenant={req.tenant_id}", flush=True)
    messages.append({"role": "user", "content":
        "У тебя закончился лимит действий на этот ход, а разбор не завершён. Не изображай "
        "законченный ответ — прямо скажи человеку, что сообщение было большое/насыщенное и ты "
        "не успел разобрать всё за один раз. Кратко резюмируй, что успел заметить, и предложи "
        "продолжить (например: чтобы дописать самому в следующем сообщении, или разбить на части)."})
    final = await _client.chat.completions.create(
        model=model_id,
        messages=_with_cache_control(messages),
        max_tokens=MAX_OUTPUT_TOKENS,
        extra_body=_EXTRA_BODY,
    )
    text = final.choices[0].message.content or "(не удалось свести ответ)"
    messages.append({"role": "assistant", "content": text})
    supervision.capture_async(req.tenant_id, req.message, text, skill=skill_used,
                              tools_used=used_tools, budget_exhausted=True)
    yield _sse("answer", {"text": text})


@app.post("/session")
async def session(req: SessionRequest):
    return StreamingResponse(_run(req), media_type="text/event-stream")


@app.get("/health")
async def health():
    return {"status": "ok", "active": _active, "sessions": len(_sessions)}


@app.delete("/session/{session_id}")
async def drop_session(session_id: str):
    """Forget a session's in-RAM conversation. The bot calls this on
    /delete_my_data so a wipe clears live context too — otherwise the coach
    keeps 'remembering' the user (and could re-save) from session history
    until TTL, even though the on-disk .md files are already gone."""
    return {"dropped": _sessions.pop(session_id, None) is not None}


# Minimal same-origin dev chat — lets Denis exercise разборы текстом in a browser
# before the TG voice handler (M2) exists. Not a product UI, just a probe.
_CHAT_HTML = """<!doctype html><html lang=ru><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>AICOACH — dev chat</title>
<style>
 body{font:16px/1.5 -apple-system,system-ui,sans-serif;max-width:760px;margin:0 auto;padding:16px;background:#0f1115;color:#e6e6e6}
 #log{white-space:pre-wrap}
 .u{color:#7db3ff;margin:18px 0 6px;font-weight:600}
 .a{background:#1a1d24;border-radius:10px;padding:12px 14px;margin:6px 0 4px}
 .meta{color:#6b7280;font-size:13px;margin:2px 0}
 textarea{width:100%;box-sizing:border-box;background:#1a1d24;color:#e6e6e6;border:1px solid #2a2f3a;border-radius:10px;padding:10px;font:inherit}
 button{margin-top:8px;padding:10px 18px;border:0;border-radius:10px;background:#2563eb;color:#fff;font:inherit;cursor:pointer}
 button:disabled{opacity:.5}
</style>
<h2>AICOACH — dev chat <span class=meta>tenant: default</span></h2>
<div id=log></div>
<textarea id=inp rows=3 placeholder="Напиши, что происходит…"></textarea><br>
<button id=send>Отправить</button>
<script>
const log=document.getElementById('log'),inp=document.getElementById('inp'),send=document.getElementById('send');
const sid='web-'+Math.random().toString(36).slice(2,8);
function add(cls,txt){const d=document.createElement('div');d.className=cls;d.textContent=txt;log.appendChild(d);d.scrollIntoView();return d}
async function ask(){
 const msg=inp.value.trim();if(!msg)return;inp.value='';send.disabled=true;
 add('u','Ты: '+msg);
 const r=await fetch('/session',{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify({tenant_id:'default',session_id:sid,message:msg})});
 const rd=r.body.getReader(),dec=new TextDecoder();let buf='';
 for(;;){const{value,done}=await rd.read();if(done)break;buf+=dec.decode(value,{stream:true});
  let i;while((i=buf.indexOf('\\n\\n'))>=0){const blk=buf.slice(0,i);buf=buf.slice(i+2);
   const ev=(blk.match(/event: (.*)/)||[])[1],dm=blk.match(/data: (.*)/s);if(!dm)continue;
   const d=JSON.parse(dm[1]);
   if(ev==='tool_call')add('meta','↳ '+d.name+'('+JSON.stringify(d.args)+')');
   else if(ev==='thought')add('meta','💭 '+d.text);
   else if(ev==='answer')add('a',d.text);
  }}
 send.disabled=false;inp.focus();
}
send.onclick=ask;
inp.addEventListener('keydown',e=>{if(e.key==='Enter'&&(e.metaKey||e.ctrlKey))ask()});
</script></html>"""


@app.get("/", response_class=HTMLResponse)
async def chat_page():
    return _CHAT_HTML

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
import time

import tiktoken
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from openai import AsyncOpenAI
from pydantic import BaseModel

import config
import memory
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


def _evict_stale():
    now = time.time()
    for sid in [s for s, v in _sessions.items() if now - v.last_used > SESSION_TTL_SECONDS]:
        del _sessions[sid]


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
            yield _sse("answer", {"text": message.content or ""})
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
            yield _sse("tool_call", {"name": name, "args": args})
            yield _sse("tool_result", {"name": name, "ok": not (isinstance(result, dict) and result.get("error"))})
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result, ensure_ascii=False, default=str),
            })

    # step budget spent — force a final answer with no tools
    messages.append({"role": "user", "content": "Заверши разбор ответом пользователю, без вызова инструментов."})
    final = await _client.chat.completions.create(
        model=model_id,
        messages=_with_cache_control(messages),
        max_tokens=MAX_OUTPUT_TOKENS,
        extra_body=_EXTRA_BODY,
    )
    text = final.choices[0].message.content or "(не удалось свести ответ)"
    messages.append({"role": "assistant", "content": text})
    yield _sse("answer", {"text": text})


@app.post("/session")
async def session(req: SessionRequest):
    return StreamingResponse(_run(req), media_type="text/event-stream")


@app.get("/health")
async def health():
    return {"status": "ok"}


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

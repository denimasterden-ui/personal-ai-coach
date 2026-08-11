"""AICOACH Telegram bot — two modes.

Private (default, PUBLIC_MODE unset): strict chat_id whitelist, single shared
TENANT_ID. Until ALLOWED_CHAT_ID is set, the bot refuses everyone and just
tells you your chat_id, so nobody else's messages ever reach the brain or the
personal profile.

Public demo (PUBLIC_MODE=true): open to any Telegram user, no whitelist. Each
chat gets its own memory, keyed by a salted hash of its chat_id (not the raw
id) — see memory.py's docstring for the honest threat model this buys. A
simple per-day message cap guards the operator's LLM budget from abuse, since
every message is a real API call on the operator's key.

Long-polling (no public URL needed). Handles text and voice: voice → Groq
whisper-large-v3 → text → POST /session (the coaching brain) → stream answer.
"""

import asyncio
import hashlib
import json
import os
import signal
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx

import analytics  # engagement analytics — hashed tenant, no message content
import memory  # direct file-level access for /memory, /export, /delete_my_data
               # (same TENANTS_DIR + encryption as the service; no HTTP hop needed)

TOKEN = os.environ["TG_BOT_TOKEN"]
API = f"https://api.telegram.org/bot{TOKEN}"
FILE_API = f"https://api.telegram.org/file/bot{TOKEN}"
AICOACH_URL = os.environ.get("AICOACH_URL", "http://127.0.0.1:8091")
TENANT_ID = os.environ.get("TENANT_ID", "default")
ALLOWED_CHAT_ID = os.environ.get("ALLOWED_CHAT_ID", "").strip()

PUBLIC_MODE = os.environ.get("PUBLIC_MODE", "false").strip().lower() == "true"
TENANT_SALT = os.environ.get("TENANT_SALT", "")
DAILY_MESSAGE_LIMIT = int(os.environ.get("DAILY_MESSAGE_LIMIT", "30"))

if PUBLIC_MODE and not TENANT_SALT:
    raise SystemExit("PUBLIC_MODE=true requires TENANT_SALT (random string) set in .env")

# The operator's own chat, if this instance is shared with public users (dogfooding
# the same deployment instead of running a second one): exempt from the daily cap,
# and optionally routed to a better/pricier model than the public default.
ADMIN_CHAT_ID = os.environ.get("ADMIN_CHAT_ID", "").strip()
ADMIN_MODEL = os.environ.get("ADMIN_MODEL", "").strip() or None


def _is_admin(chat_id: int) -> bool:
    return bool(ADMIN_CHAT_ID) and str(chat_id) == ADMIN_CHAT_ID


def _tenant_id_for(chat_id: int) -> str:
    """Private mode: one shared TENANT_ID (as before). Public mode: each chat
    gets its own tenant, named by a salted hash of chat_id rather than the raw
    id, so a directory listing of tenants/ doesn't read as a Telegram roster."""
    if not PUBLIC_MODE:
        return TENANT_ID
    return hashlib.sha256(f"{TENANT_SALT}:{chat_id}".encode()).hexdigest()[:24]


# Per-day message cap per chat, public mode only — caps worst-case API cost
# from a single abusive/looping user. In-memory (resets on restart), fine for
# a soft guard; not meant as a hard security boundary.
_daily_count: dict[int, tuple[int, int]] = {}  # chat_id -> (day_number, count)


def _rate_limited(chat_id: int) -> bool:
    if not PUBLIC_MODE or _is_admin(chat_id):
        return False
    day = int(time.time() // 86400)
    d, n = _daily_count.get(chat_id, (day, 0))
    if d != day:
        d, n = day, 0
    n += 1
    _daily_count[chat_id] = (d, n)
    limited = n > DAILY_MESSAGE_LIMIT
    if limited:
        analytics.log("rate_limited", _tenant_id_for(chat_id))
    return limited


# ── weekly opt-in nudge ───────────────────────────────────────────────────────
# Opt-in ("/weekly on"): a week after the last check-in the bot pings you to come
# back and reflect — the retention loop that makes long-term memory pay off. The
# push needs the raw chat_id (Telegram addresses by it), so subscribers persist
# to a small file; it's encrypted with the same key as memory when one is set, so
# in public mode this file doesn't become a plaintext Telegram roster either.
WEEKLY_INTERVAL = 7 * 86400
SUBSCRIBERS_FILE = Path(os.environ.get("SUBSCRIBERS_FILE", "subscribers.enc"))

_sub_fernet = None
if os.environ.get("MEMORY_ENCRYPTION_KEY", ""):
    from cryptography.fernet import Fernet
    _sub_fernet = Fernet(os.environ["MEMORY_ENCRYPTION_KEY"].encode())

_subs: dict[str, float] = {}  # str(chat_id) -> last_push_epoch


def _load_subs() -> None:
    if not SUBSCRIBERS_FILE.exists():
        return
    try:
        data = SUBSCRIBERS_FILE.read_bytes()
        if _sub_fernet:
            data = _sub_fernet.decrypt(data)
        _subs.update(json.loads(data.decode("utf-8")))
    except Exception as exc:
        print("[weekly] load error:", exc, flush=True)


def _save_subs() -> None:
    try:
        data = json.dumps(_subs).encode("utf-8")
        if _sub_fernet:
            data = _sub_fernet.encrypt(data)
        SUBSCRIBERS_FILE.write_bytes(data)
    except Exception as exc:
        print("[weekly] save error:", exc, flush=True)


WEEKLY_NUDGE = (
    "Прошла неделя. Как ты?\n\n"
    "Что изменилось с прошлого раза — в мыслях, в решениях, в том, что беспокоило? "
    "Наговори или напиши, я помню, на чём мы остановились.\n\n"
    "(Отключить напоминания — /weekly off)"
)


async def _weekly_loop(client):
    while True:
        try:
            now = time.time()
            due = [cid for cid, last in list(_subs.items()) if now - last >= WEEKLY_INTERVAL]
            for cid in due:
                r = await _tg(client, "sendMessage", chat_id=int(cid), text=WEEKLY_NUDGE)
                if r.get("ok"):
                    _subs[cid] = now
                    analytics.log("weekly_push", _tenant_id_for(int(cid)))
                    print(f"[weekly] pushed to {cid}", flush=True)
                elif not r.get("ok") and r.get("error_code") == 403:
                    _subs.pop(cid, None)  # user blocked the bot — drop them
                    analytics.log("blocked", _tenant_id_for(int(cid)))
                    print(f"[weekly] {cid} blocked bot, unsubscribed", flush=True)
            if due:
                _save_subs()
        except Exception as exc:
            print("[weekly] loop error:", exc, flush=True)
        await asyncio.sleep(3600)


# STT via Groq (whisper-large-v3): fast on a GPU-less VPS. Groq accepts the
# Telegram .oga (ogg/opus) directly, so no ffmpeg step. Trade-off (accepted):
# voice leaves to Groq — see R2 in TZ.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_STT_MODEL = os.environ.get("GROQ_STT_MODEL", "whisper-large-v3")


async def _transcribe(client, audio: bytes) -> str:
    r = await client.post(
        "https://api.groq.com/openai/v1/audio/transcriptions",
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        files={"file": ("voice.ogg", audio, "audio/ogg")},
        data={"model": GROQ_STT_MODEL, "language": "ru"},
        timeout=60,
    )
    if r.status_code != 200:
        print(f"[stt] groq error {r.status_code}: {r.text[:200]}", flush=True)
        return ""
    return (r.json().get("text") or "").strip()


# PDF attachments arrive as material for the current turn — never as a skill.
# Skills are a shared, cross-tenant resource read straight into every system
# prompt, so a user-supplied file must not be able to reach them (ADR 0003
# threat model). Extracted text goes into the turn like any other message: what
# survives is only what the coach itself chooses to save_memory.
DOC_MAX_BYTES = 20 * 1024 * 1024  # Telegram's getFile ceiling — bigger never downloads
DOC_MAX_CHARS = 30000             # one attachment must not eat the whole context

# Plain text needs no parser — a transcript or exported notes are as legitimate a
# source for a profile claim as a PDF report, and refusing them sent the user
# back to copy-pasting. Telegram's mime for .md is unreliable (text/markdown,
# application/octet-stream, sometimes absent), so the suffix decides too.
TEXT_SUFFIXES = (".md", ".markdown", ".txt")


def _doc_kind(name: str, mime: str | None) -> str | None:
    low = name.lower()
    if mime == "application/pdf" or low.endswith(".pdf"):
        return "pdf"
    if (mime or "").startswith("text/") or low.endswith(TEXT_SUFFIXES):
        return "text"
    return None


def _pdf_text_sync(data: bytes) -> str:
    from io import BytesIO

    from pypdf import PdfReader

    pages = PdfReader(BytesIO(data)).pages
    return "\n\n".join((p.extract_text() or "") for p in pages).strip()


async def _pdf_text(data: bytes) -> str:
    """Text layer of a PDF, or '' if there is none (a scan) or it won't parse.
    Runs off the loop — pypdf is blocking and a big file would stall every chat."""
    try:
        return await asyncio.to_thread(_pdf_text_sync, data)
    except Exception as exc:
        print(f"[pdf] parse error: {exc}", flush=True)
        return ""


async def _tg(client, method, **params):
    r = await client.post(f"{API}/{method}", json=params, timeout=30)
    return r.json()


async def _react(client, chat_id, message_id, emoji="👀"):
    """Put a reaction on an incoming message so the user sees it was read —
    instant feedback even while the answer is still cooking. Best-effort."""
    try:
        await _tg(client, "setMessageReaction", chat_id=chat_id, message_id=message_id,
                  reaction=[{"type": "emoji", "emoji": emoji}])
    except Exception:
        pass


TG_MAX = 4096  # Telegram's hard per-message limit; longer sends 400 and vanishes


async def _send_text(client, chat_id, text, **kw):
    """Split text over 4096 chars on line boundaries and send as several messages —
    a rich /memory profile or a long coach answer would otherwise silently 400."""
    if not text:
        return
    while text:
        if len(text) <= TG_MAX:
            chunk, text = text, ""
        else:
            cut = text.rfind("\n", 0, TG_MAX)
            if cut < TG_MAX // 2:
                cut = TG_MAX
            chunk, text = text[:cut], text[cut:].lstrip("\n")
        await _tg(client, "sendMessage", chat_id=chat_id, text=chunk, **kw)


async def _ask_brain(client, tenant_id, session_id, text, model=None, turn_id=None) -> str:
    """POST /session, consume the SSE stream, return the final answer text.
    Also logs a memory_write analytics event per save_memory tool call — a
    cheap proxy for "the coach actually learned something this turn"."""
    answer = ""
    payload = {"tenant_id": tenant_id, "session_id": session_id, "message": text}
    if model:
        payload["model"] = model
    if turn_id:
        payload["turn_id"] = turn_id
    async with client.stream(
        "POST", f"{AICOACH_URL}/session", json=payload, timeout=300,
    ) as resp:
        event = None
        async for line in resp.aiter_lines():
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
                if event == "answer":
                    answer = data.get("text", "")
                # memory_write analytics removed (aicoach-dev#11): after #10 moved
                # memory writes to the recollect pass, the coach no longer calls
                # save_memory, so this listener never fires. Memory write stats
                # now come from supervision.pass_stats() instead.
    return answer or "(пустой ответ)"


# Per-chat serial worker (the "followup" queue pattern). A person often sends a
# thought as several messages in a row — and may keep talking while the brain is
# still answering the previous batch. One worker task per chat guarantees exactly
# one brain call for that chat at a time: no two turns ever mutate the same
# session concurrently (which used to interleave context and yield stub answers).
# Messages that arrive mid-answer aren't dropped or run in parallel — they land
# in the buffer and become the next turn, with the previous answer already in
# context (so the coach builds on it, like a real conversation).
DEBOUNCE_SEC = float(os.environ.get("DEBOUNCE_SEC", "4"))
_chat: dict[int, dict] = {}  # chat_id -> {"parts": [str], "voice": bool, "last": float, "worker": Task|None}

# In-flight per-chat workers. On SIGTERM we stop taking new updates and await
# these so a deploy never kills a разбор mid-answer.
_inflight: set[asyncio.Task] = set()
DRAIN_TIMEOUT = float(os.environ.get("DRAIN_TIMEOUT", "85"))  # < systemd TimeoutStopSec


LONG_INPUT_CHARS = 2000


async def _typing_keepalive(client, chat_id):
    """Telegram's 'typing' status expires after ~5s — refresh it so the whole
    (possibly minutes-long) brain call reads as active, not frozen."""
    try:
        while True:
            await _tg(client, "sendChatAction", chat_id=chat_id, action="typing")
            await asyncio.sleep(4)
    except asyncio.CancelledError:
        pass


async def _worker(client, chat_id):
    """Drain a chat's buffer one turn at a time until it's empty. Loops so that
    anything that arrives while a turn is being answered becomes the next turn."""
    st = _chat[chat_id]
    while st["parts"]:
        # debounce: wait until DEBOUNCE_SEC of quiet since the last message, so a
        # burst of fragments joins into one turn. `preparing` holds the window
        # open while an attachment is still being fetched and parsed — a PDF takes
        # seconds, and without this the text sent alongside it flushed alone and
        # got its own separate answer.
        while True:
            if st["preparing"]:
                await asyncio.sleep(0.2)
                continue
            gap = DEBOUNCE_SEC - (time.time() - st["last"])
            if gap <= 0:
                break
            await asyncio.sleep(gap)
        text = "\n\n".join(st["parts"]); st["parts"] = []
        voice = st["voice"]; st["voice"] = False
        tenant = _tenant_id_for(chat_id)
        turn_id = uuid.uuid4().hex
        print(f"[flush] chat={chat_id} chars={len(text)} turn_id={turn_id}", flush=True)
        analytics.log("message", tenant, kind="voice" if voice else "text")
        if len(text) > LONG_INPUT_CHARS:
            await _tg(client, "sendMessage", chat_id=chat_id,
                      text="Принял, обрабатываю — большой объём, это займёт пару минут…")
        model = ADMIN_MODEL if _is_admin(chat_id) else None
        ka = asyncio.create_task(_typing_keepalive(client, chat_id))
        try:
            answer = await _ask_brain(client, tenant, f"tg-{chat_id}", text, model=model,
                                      turn_id=turn_id)
        except Exception as exc:
            # brain unreachable (e.g. mid-deploy restart) — don't leave the user in silence
            print("[flush] brain error:", exc, flush=True)
            await _send_text(client, chat_id,
                             "Секунду — я на пару мгновений обновился. Повтори, пожалуйста, "
                             "последнее сообщение, я на месте.")
            continue
        finally:
            ka.cancel()
        await _send_text(client, chat_id, answer)
        analytics.log("answer", tenant)
        print(f"[answer] sent {len(answer)} chars", flush=True)


def _state(chat_id):
    return _chat.setdefault(chat_id, {"parts": [], "voice": False, "last": 0.0,
                                      "worker": None, "preparing": 0})


def _buffer(client, chat_id, text, kind="text"):
    st = _state(chat_id)
    st["parts"].append(text)
    st["voice"] = st["voice"] or kind == "voice"
    st["last"] = time.time()
    w = st["worker"]
    if w is None or w.done():
        w = asyncio.create_task(_worker(client, chat_id))
        st["worker"] = w
        _inflight.add(w)
        w.add_done_callback(_inflight.discard)


def _onboarding(chat_id: int) -> str:
    base = (
        "🧭 Я — AI-коуч с долговременной памятью.\n\n"
        "Наговори или напиши, что происходит — помогу отделить факты, эмоции, "
        "паттерны и следующий шаг. Работаю по интегративным методикам (IFS, CBT, "
        "ACT и др.). Я не заменяю психотерапевта и не оказываю мед. помощь.\n\n"
        "Чем отличаюсь от обычного чата: помню твою историю между разговорами и "
        "постепенно строю твой профиль. Вернёшься через неделю — вспомню, на чём "
        "мы остановились.\n\n"
        "Команды:\n"
        "• /memory — что я о тебе понял\n"
        "• /export — забрать свою память файлом .md\n"
        "• /weekly — напоминание раз в неделю вернуться к рефлексии\n"
        "• /delete_my_data — стереть всё о тебе"
    )
    if PUBLIC_MODE and not _is_admin(chat_id):
        base += (
            f"\n\n🔒 Это публичное демо (лимит {DAILY_MESSAGE_LIMIT} сообщений/день). "
            "Память изолирована и шифруется на сервере, но оператор технически "
            "имеет доступ к ключу — не пиши того, что не готов ему доверить. Для "
            "полной приватности разверни свой инстанс:\n"
            "github.com/denimasterden-ui/personal-ai-coach"
        )
    return base + "\n\nНачнём? Расскажи, что сейчас занимает."


async def _send_document(client, chat_id, filename, content):
    await client.post(
        f"{API}/sendDocument",
        data={"chat_id": chat_id},
        files={"document": (filename, content.encode("utf-8"), "text/markdown")},
        timeout=30,
    )


async def _command(client, chat_id, text):
    cmd, _, arg = text.partition(" ")
    cmd, arg = cmd.lower(), arg.strip().lower()
    tenant = _tenant_id_for(chat_id)

    if not analytics.has_seen(tenant):
        # deep-link attribution: t.me/<bot>?start=reddit → "/start reddit" is
        # always the very first message Telegram sends, so real /start carries
        # the source; anyone who starts by typing something else gets "direct".
        analytics.log("first_seen", tenant, source=arg if cmd == "/start" else "")
    analytics.log("command", tenant, cmd=cmd)

    if cmd == "/start":
        await _tg(client, "sendMessage", chat_id=chat_id, text=_onboarding(chat_id))
    elif cmd == "/memory":
        summary = await memory.summarize(tenant)
        await _send_text(client, chat_id,
                         summary or "Пока пусто — расскажи о себе, и я начну запоминать.")
    elif cmd == "/export":
        dump = await memory.export_tenant(tenant)
        if not dump:
            await _tg(client, "sendMessage", chat_id=chat_id, text="Пока нечего экспортировать.")
        else:
            await _send_document(client, chat_id, "aicoach-memory.md", dump)
    elif cmd == "/delete_my_data":
        # Telegram auto-links /commands in message text, so /delete_confirm below
        # renders as a single tappable confirmation — no typed argument needed.
        await _tg(client, "sendMessage", chat_id=chat_id,
                  text="Это удалит ВСЮ память о тебе безвозвратно. Сначала можешь "
                       "забрать её через /export.\n\nЧтобы подтвердить — нажми /delete_confirm")
    elif cmd == "/delete_confirm":
        ok = await memory.delete_tenant(tenant)
        # also drop the live in-RAM conversation in the service, else the coach
        # keeps "remembering" from session context (and may re-save) until TTL.
        try:
            await client.delete(f"{AICOACH_URL}/session/tg-{chat_id}", timeout=10)
        except Exception as exc:
            print("[delete] session reset failed:", exc, flush=True)
        if _subs.pop(str(chat_id), None) is not None:
            _save_subs()
        analytics.log("deleted", tenant)
        await _tg(client, "sendMessage", chat_id=chat_id,
                  text="Готово — вся твоя память удалена." if ok else "Данных и так не было.")
    elif cmd == "/weekly":
        if arg == "on":
            _subs[str(chat_id)] = time.time()
            _save_subs()
            analytics.log("weekly_on", tenant)
            await _tg(client, "sendMessage", chat_id=chat_id,
                      text="Включил. Через неделю напомню вернуться к рефлексии. Отключить — /weekly off")
        elif arg == "off":
            if _subs.pop(str(chat_id), None) is not None:
                _save_subs()
            analytics.log("weekly_off", tenant)
            await _tg(client, "sendMessage", chat_id=chat_id, text="Напоминания отключены.")
        else:
            on = str(chat_id) in _subs
            await _tg(client, "sendMessage", chat_id=chat_id,
                      text=f"Еженедельные напоминания: {'включены' if on else 'выключены'}.\n"
                           "Включить — /weekly on · Выключить — /weekly off")
    elif cmd == "/stats" and _is_admin(chat_id):
        days = int(arg) if arg.isdigit() else 30
        await _send_text(client, chat_id, analytics.format_summary(analytics.summary(days)))
    else:
        await _tg(client, "sendMessage", chat_id=chat_id,
                  text="Неизвестная команда. /start — список команд.")


# Registered with Telegram so typing "/" shows a command menu. /delete_confirm is
# deliberately omitted — it's only reachable via the /delete_my_data warning, so it
# can't be tapped straight from the menu and wipe data by accident.
_COMMANDS = [
    {"command": "start", "description": "О боте и список команд"},
    {"command": "memory", "description": "Что я о тебе понял"},
    {"command": "export", "description": "Забрать свою память файлом .md"},
    {"command": "weekly", "description": "Еженедельное напоминание вернуться к рефлексии"},
    {"command": "delete_my_data", "description": "Стереть всё о тебе"},
]


async def _set_commands(client):
    await _tg(client, "setMyCommands", commands=_COMMANDS)


async def _handle(client, msg):
    """Keep the debounce window open while a slow attachment is prepared. The
    counter must be raised before the first await, so the worker started by a
    text message that arrived in the same batch can't flush ahead of it."""
    slow = bool({"voice", "audio", "document"} & msg.keys())
    st = _state(msg["chat"]["id"]) if slow else None
    if st:
        st["preparing"] += 1
    try:
        await _handle_inner(client, msg)
    finally:
        if st:
            st["preparing"] -= 1


async def _handle_inner(client, msg):
    chat_id = msg["chat"]["id"]
    kind = ("voice" if ("voice" in msg or "audio" in msg)
            else "document" if "document" in msg
            else "text" if "text" in msg else "other")
    print(f"[msg] chat={chat_id} kind={kind}", flush=True)

    if PUBLIC_MODE:
        if _rate_limited(chat_id):
            await _tg(client, "sendMessage", chat_id=chat_id,
                      text=f"Дневной лимит сообщений ({DAILY_MESSAGE_LIMIT}) исчерпан — приходи завтра.")
            return
    else:
        if not ALLOWED_CHAT_ID:
            await _tg(client, "sendMessage", chat_id=chat_id,
                      text=f"Твой chat_id: {chat_id}\nДобавь его в ALLOWED_CHAT_ID и перезапусти бота.")
            return
        if str(chat_id) != ALLOWED_CHAT_ID:
            await _tg(client, "sendMessage", chat_id=chat_id, text="Нет доступа.")
            return

    if "voice" in msg or "audio" in msg:
        if not GROQ_API_KEY:
            await _tg(client, "sendMessage", chat_id=chat_id,
                      text="🎙 Голосовые на этом боте пока не поддержаны — напиши, пожалуйста, текстом.")
            return
        await _react(client, chat_id, msg["message_id"])  # 👀 «услышал, расшифровываю»
        await _tg(client, "sendChatAction", chat_id=chat_id, action="typing")
        file_id = (msg.get("voice") or msg.get("audio"))["file_id"]
        info = await _tg(client, "getFile", file_id=file_id)
        file_path = info["result"]["file_path"]
        audio = (await client.get(f"{FILE_API}/{file_path}", timeout=60)).content
        text = await _transcribe(client, audio)
        print(f"[stt] {text[:120]!r}", flush=True)
        if not text:
            await _tg(client, "sendMessage", chat_id=chat_id, text="Не разобрал голосовое, повтори?")
            return
        # echo recognition immediately (feedback), buffer for the joined turn
        await _tg(client, "sendMessage", chat_id=chat_id, text=f"🎙 _{text}_", parse_mode="Markdown")
        if not analytics.has_seen(_tenant_id_for(chat_id)):
            analytics.log("first_seen", _tenant_id_for(chat_id), source="")
        _buffer(client, chat_id, text, kind="voice")
    elif "text" in msg:
        text = msg["text"]
        if text.startswith("/"):
            await _command(client, chat_id, text)
            return
        await _react(client, chat_id, msg["message_id"])  # 👀 «прочитал»
        if not analytics.has_seen(_tenant_id_for(chat_id)):
            analytics.log("first_seen", _tenant_id_for(chat_id), source="")
        _buffer(client, chat_id, text)
    elif "document" in msg:
        doc = msg["document"]
        name = doc.get("file_name") or "документ"
        kind_doc = _doc_kind(name, doc.get("mime_type"))
        if kind_doc is None:
            print(f"[doc] отказ: {name!r} тип {doc.get('mime_type')!r} не поддержан", flush=True)
            await _tg(client, "sendMessage", chat_id=chat_id,
                      text=f"Из файлов читаю PDF и текстовые (.md, .txt) — «{name}» не возьму. "
                           "Пришли, пожалуйста, текстом или голосовым.")
            return
        if (doc.get("file_size") or 0) > DOC_MAX_BYTES:
            print(f"[doc] отказ: {name!r} {doc.get('file_size')}B больше потолка", flush=True)
            await _tg(client, "sendMessage", chat_id=chat_id,
                      text="Файл больше 20 МБ — Telegram такой боту не отдаёт. "
                           "Пришли часть текстом.")
            return
        await _react(client, chat_id, msg["message_id"])  # 👀 «взял в работу»
        await _tg(client, "sendChatAction", chat_id=chat_id, action="typing")
        info = await _tg(client, "getFile", file_id=doc["file_id"])
        file_path = (info.get("result") or {}).get("file_path")
        if not file_path:
            print(f"[doc] отказ: getFile не отдал файл — {str(info)[:200]}", flush=True)
            await _tg(client, "sendMessage", chat_id=chat_id,
                      text="Не смог забрать файл у Telegram. Попробуй прислать ещё раз?")
            return
        data = (await client.get(f"{FILE_API}/{file_path}", timeout=120)).content
        if kind_doc == "pdf":
            text = await _pdf_text(data)
            if not text:
                print(f"[doc] отказ: {name!r} без текстового слоя (скан?)", flush=True)
                await _tg(client, "sendMessage", chat_id=chat_id,
                          text=f"В «{name}» нет текстового слоя — похоже, это скан. "
                               "Распознавать картинки я пока не умею, пришли текстом.")
                return
        else:
            try:
                text = data.decode("utf-8").strip()
            except UnicodeDecodeError:
                text = ""
            if not text:
                print(f"[doc] отказ: {name!r} не читается как UTF-8", flush=True)
                await _tg(client, "sendMessage", chat_id=chat_id,
                          text=f"«{name}» не читается как текст в UTF-8. "
                               "Пересохрани в UTF-8 или пришли содержимое сообщением.")
                return
        full_chars = len(text)
        tenant = _tenant_id_for(chat_id)
        # Сохраняем распознанный текст ЦЕЛИКОМ как память типа doc — обрезка
        # ниже касается только того, что уходит в реплику хода, не самого
        # документа. По slug коуч достаёт полный текст через load_doc; recall
        # отдаёт лишь превью, так что большой отчёт не затапливает контекст.
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        slug = memory._slug(name)  # тот же slug, по которому память положит файл
        source = f"файл «{name}», прислан в Telegram {stamp}"
        saved = await memory.save_memory(tenant, "doc", text, slug=slug, source=source)
        # save_memory не бросает, а возвращает {"error": ...}. Без этой проверки
        # сбой записи выглядел бы как успех: пользователю «сохранил целиком», а
        # коучу — load_doc(slug), который ничего не найдёт.
        stored = not (isinstance(saved, dict) and saved.get("error"))
        if not stored:
            print(f"[doc] сохранить {slug!r} не удалось: {saved}", flush=True)
        # В реплику хода идёт лишь превью — полный текст лежит в памяти и
        # доступен по slug. Превью помечаем обрезкой явно, чтобы коуч не
        # рассуждал по огрызку как по целому документу.
        clipped = full_chars > DOC_MAX_CHARS
        preview = text[:DOC_MAX_CHARS]
        print(f"[doc] {kind_doc} {name!r} {len(data)}B -> {full_chars} симв. сохранено, "
              f"в ход {len(preview)}, обрезано={clipped}", flush=True)
        await _tg(client, "sendMessage", chat_id=chat_id,
                  text=(f"📄 «{name}» — прочитал {full_chars} символов и сохранил целиком."
                        + (f" В ход идёт только начало ({DOC_MAX_CHARS} символов), "
                           "остальное коуч дочитает сам." if clipped else "")
                        if stored else
                        f"📄 «{name}» — прочитал {full_chars} символов, но сохранить не смог. "
                        "В разбор уйдёт только начало документа."))
        if not analytics.has_seen(tenant):
            analytics.log("first_seen", tenant, source="")
        # Шапка обязана быть честной в обе стороны: что документ есть целиком и как
        # его дочитать — либо что ниже огрызок и остального не будет.
        head = (f"[Документ «{name}» ({full_chars} символов) сохранён целиком, "
                f"полный текст: load_doc(\"{slug}\")."
                + (f" Ниже только первые {DOC_MAX_CHARS} символов." if clipped else "")
                if stored else
                f"[Документ «{name}» ({full_chars} символов) сохранить не удалось, "
                f"дочитать нечем. Ниже "
                + (f"только первые {DOC_MAX_CHARS} символов — остального нет."
                   if clipped else "весь распознанный текст."))
        _buffer(client, chat_id, f"{head}]\n\n{preview}")
    else:
        await _tg(client, "sendMessage", chat_id=chat_id,
                  text="Я понимаю текст, голосовые и PDF. Это пока не возьму.")


async def main():
    offset = None
    mode = f"PUBLIC (hashed tenants, {DAILY_MESSAGE_LIMIT}/day cap)" if PUBLIC_MODE else (
        "private, whitelist=set" if ALLOWED_CHAT_ID else "private, whitelist=OPEN (reveals chat_id)")
    _load_subs()
    analytics.init()
    print(f"AICOACH bot up. mode={mode}, weekly_subs={len(_subs)}")

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            pass

    async with httpx.AsyncClient() as client:
        await _set_commands(client)
        asyncio.create_task(_weekly_loop(client))
        while not stop.is_set():
            # race the long-poll against SIGTERM so shutdown is prompt
            poll = asyncio.ensure_future(_tg(client, "getUpdates", offset=offset, timeout=25))
            stopw = asyncio.ensure_future(stop.wait())
            await asyncio.wait({poll, stopw}, return_when=asyncio.FIRST_COMPLETED)
            if stop.is_set():
                poll.cancel()
                break
            stopw.cancel()
            try:
                upd = poll.result()
            except Exception as exc:
                print("getUpdates error:", exc)
                await asyncio.sleep(3)
                continue
            for u in upd.get("result", []):
                offset = u["update_id"] + 1
                msg = u.get("message")
                if msg:
                    try:
                        await _handle(client, msg)
                    except Exception as exc:
                        print("handle error:", exc)

        print(f"[shutdown] draining {len(_inflight)} in-flight turn(s)…", flush=True)
        if _inflight:
            await asyncio.wait(set(_inflight), timeout=DRAIN_TIMEOUT)
        print("[shutdown] done", flush=True)


if __name__ == "__main__":
    asyncio.run(main())

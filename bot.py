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
import re
import signal
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx

import analytics  # engagement analytics — hashed tenant, no message content
import memory  # direct file-level access for /memory, /export, /delete_my_data
               # (same TENANTS_DIR + encryption as the service; no HTTP hop needed)
import supervision

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


# ── product store (ADR 0004): chat_id + tour state ────────────────────────────
# The raw chat_id of everyone who pressed /start. The tenant hash is one-way, so
# without this store the operator can never reach a person again once the in-RAM
# session TTLs — eleven users were already lost this way. Encrypted with the same
# key as subscriptions; a leaked file must not read as a plaintext Telegram roster.
# The value reserves room for tour state (onboarding progress), not built out yet.
PRODUCT_STORE_FILE = Path(os.environ.get("PRODUCT_STORE_FILE", "contacts.enc"))

_contacts: dict[str, dict] = {}  # str(chat_id) -> {"tour": ...}  (tour state TBD)


def _load_contacts() -> None:
    if not PRODUCT_STORE_FILE.exists():
        return
    try:
        data = PRODUCT_STORE_FILE.read_bytes()
        if _sub_fernet:
            data = _sub_fernet.decrypt(data)
        _contacts.update(json.loads(data.decode("utf-8")))
    except Exception as exc:
        print("[contacts] load error:", exc, flush=True)


def _save_contacts() -> None:
    try:
        data = json.dumps(_contacts).encode("utf-8")
        if _sub_fernet:
            data = _sub_fernet.encrypt(data)
        PRODUCT_STORE_FILE.write_bytes(data)
    except Exception as exc:
        print("[contacts] save error:", exc, flush=True)


# ── product feedback (aicoach-dev#20) ────────────────────────────────────────
# After FEEDBACK_AFTER_TURN completed turns the bot — not the coach — asks once,
# as a separate message, whether the product was useful. The answer is about the
# product, not about the person, so it must never enter the brain as a turn or
# reach the extraction pass (ADR 0004 rule 1). Asked exactly once: the
# feedback_asked marker persists in the product store; on restart the person who
# hasn't answered yet just continues talking normally.
FEEDBACK_AFTER_TURN = 3

FEEDBACK_QUESTION = (
    "Как тебе вообще — полезно ли?\n\n"
    "Расскажи своими словами, что зашло, а что нет. "
    "Можно голосовым — я расшифрую."
)

FEEDBACK_THANKS = "Спасибо за обратную связь."

# chat_ids whose next text/voice message is the feedback answer (not a turn).
# Ephemeral: a restart clears it — the question was already asked (feedback_asked
# persists), so it won't be repeated, and conversation continues without it.
_feedback_pending: set[int] = set()


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

# ── rating buttons (aicoach-dev#18) ──────────────────────────────────────────
# Accuracy axis, not sympathy: the person rates whether the разбор was on target,
# which is the only prospective signal for the hypothesis that memory improves coaching.
_RATING_LABELS = [("hit", "В точку"), ("partial", "Частично"), ("miss", "Мимо")]
_VALID_TURN_ID = re.compile(r"^[0-9a-f]{32}$")
_VALID_RATINGS = {"hit", "partial", "miss"}
_RATING_ACK = {"hit": "✓ В точку", "partial": "✓ Частично", "miss": "✓ Мимо"}


def _rating_keyboard(turn_id, selected=None):
    """Inline keyboard with three rating buttons. selected marks the current choice."""
    return {"inline_keyboard": [[
        {"text": f"{'✓ ' if v == selected else ''}{label}",
         "callback_data": f"rate:{turn_id}:{v}"}
        for v, label in _RATING_LABELS
    ]]}


# ── onboarding tour (aicoach-dev#19) ─────────────────────────────────────────
# Short step-by-step tour offered as a button on /start — not as text replacing
# a разбор. State lives in the product store (ADR 0004 rule 2): it's a product
# routing fact, not a personal fact, so the brain never sees it.
_TOUR_STEPS = [
    "🧠 Память между разговорами\n\nПосле каждого разговора я извлекаю паттерны и наблюдения — и помню их, когда ты возвращаешься.\n\n/memory — посмотреть, что накоплено.",
    "🔍 Разбор в контексте\n\nКаждое сообщение я разбираю с опорой на то, что уже знаю о тебе — не новый чат каждый раз, а коуч, который помнит историю.",
    "🔒 Данные под контролем\n\n/export — забрать всю память файлом. /delete_my_data — стереть всё безвозвратно.\n\nТеперь просто расскажи, что сейчас занимает — начнём.",
]


def _tour_offer_keyboard():
    return {"inline_keyboard": [[
        {"text": "Как это устроено?", "callback_data": "tour:0"},
    ]]}


def _tour_nav_keyboard(step: int):
    last = step == len(_TOUR_STEPS) - 1
    row = []
    if not last:
        row.append({"text": "Дальше →", "callback_data": f"tour:{step + 1}"})
    row.append({"text": "Понятно" if last else "Хватит", "callback_data": "tour:exit"})
    return {"inline_keyboard": [row]}


async def _send_text(client, chat_id, text, **kw):
    """Split text over 4096 chars on line boundaries and send as several messages —
    a rich /memory profile or a long coach answer would otherwise silently 400.
    reply_markup, if given, is attached only to the last chunk."""
    if not text:
        return
    reply_markup = kw.pop("reply_markup", None)
    while text:
        if len(text) <= TG_MAX:
            chunk, text = text, ""
        else:
            cut = text.rfind("\n", 0, TG_MAX)
            if cut < TG_MAX // 2:
                cut = TG_MAX
            chunk, text = text[:cut], text[cut:].lstrip("\n")
        extra = dict(kw)
        if not text and reply_markup is not None:  # last chunk only
            extra["reply_markup"] = reply_markup
        await _tg(client, "sendMessage", chat_id=chat_id, text=chunk, **extra)


async def _ask_brain(client, tenant_id, session_id, text, model=None, turn_id=None, light_intro=False) -> str:
    """POST /session, consume the SSE stream, return the final answer text.
    Also logs a memory_write analytics event per save_memory tool call — a
    cheap proxy for "the coach actually learned something this turn"."""
    answer = ""
    payload = {"tenant_id": tenant_id, "session_id": session_id, "message": text,
               "light_intro": light_intro}
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


async def _ask_portrait(client, tenant_id) -> dict:
    """Ask the separate portrait endpoint; it never becomes a session turn."""
    response = await client.post(
        f"{AICOACH_URL}/portrait", json={"tenant_id": tenant_id}, timeout=300,
    )
    return response.json()


async def _maybe_ask_feedback(client, chat_id):
    """Count completed turns; after the 3rd, ask for product feedback exactly
    once. The turn counter and the feedback_asked marker live in the product
    store (ADR 0004) — the brain never sees either."""
    cid = str(chat_id)
    entry = _contacts.setdefault(cid, {})
    count = entry.get("turn_count", 0) + 1
    entry["turn_count"] = count
    ask = count == FEEDBACK_AFTER_TURN and not entry.get("feedback_asked")
    if ask:
        entry["feedback_asked"] = True
        _feedback_pending.add(chat_id)
    _save_contacts()
    if ask:
        await _tg(client, "sendMessage", chat_id=chat_id, text=FEEDBACK_QUESTION)
        analytics.log("feedback_asked", _tenant_id_for(chat_id))
        print(f"[feedback] asked chat={chat_id}", flush=True)


async def _capture_feedback(client, chat_id, text):
    """Store the feedback answer and thank the person. The answer does NOT go to
    the brain and does NOT enter the extraction pass — it's a fact about the
    product (ADR 0004), kept in the product store, purged by /delete_my_data."""
    cid = str(chat_id)
    _contacts.setdefault(cid, {})["feedback"] = text
    _save_contacts()
    analytics.log("feedback", _tenant_id_for(chat_id))
    await _tg(client, "sendMessage", chat_id=chat_id, text=FEEDBACK_THANKS)
    print(f"[feedback] captured {len(text)} chars from chat={chat_id}", flush=True)


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


_GREETING_WORDS = {"привет", "здравствуйте", "добрый", "день", "вечер", "утро",
                   "hi", "hello", "hey"}


def _is_brief_first_message(text: str) -> bool:
    """Greetings should get an easy first question; a real account should go
    straight to the coach. The brain, not this small router, still handles
    safety when a brief message signals an acute state."""
    words = [word.strip(".,!?…:;").lower() for word in text.split()]
    return not words or all(word in _GREETING_WORDS for word in words)


async def _needs_light_intro(chat_id, text: str, attachment: bool) -> bool:
    _, light_intro = await _first_contact_route(chat_id, text, attachment)
    return light_intro


async def _first_contact_route(chat_id, text: str, attachment: bool) -> tuple[bool, bool]:
    seen = _contacts.get(str(chat_id), {}).get("first_contact_seen", False)
    first_contact = not seen and not await memory.has_derived_records(_tenant_id_for(chat_id))
    return first_contact, first_contact and not attachment and _is_brief_first_message(text)


async def _remember_first_contact(chat_id) -> None:
    """Make first contact durable even when recollection finds no personal fact
    (e.g. a bare "привет" that recollect extracts nothing from) — otherwise a
    week-old return with still-empty memory would be routed into the intro
    again. This is product routing state (ADR 0004), so it lives in the
    product store keyed by chat_id, not in the person's memory."""
    cid = str(chat_id)
    _contacts.setdefault(cid, {})["first_contact_seen"] = True
    _save_contacts()


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
        attachment = st["attachment"]; st["attachment"] = False
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
            first_contact, light_intro = await _first_contact_route(chat_id, text, attachment)
            if first_contact:
                await _remember_first_contact(chat_id)
            answer = await _ask_brain(client, tenant, f"tg-{chat_id}", text, model=model,
                                      turn_id=turn_id, light_intro=light_intro)
        except Exception as exc:
            # brain unreachable (e.g. mid-deploy restart) — don't leave the user in silence
            print("[flush] brain error:", exc, flush=True)
            await _send_text(client, chat_id,
                             "Секунду — я на пару мгновений обновился. Повтори, пожалуйста, "
                             "последнее сообщение, я на месте.")
            continue
        finally:
            ka.cancel()
        await _send_text(client, chat_id, answer,
                         reply_markup=_rating_keyboard(turn_id))
        analytics.log("answer", tenant)
        print(f"[answer] sent {len(answer)} chars", flush=True)
        await _maybe_ask_feedback(client, chat_id)


def _state(chat_id):
    return _chat.setdefault(chat_id, {"parts": [], "voice": False, "last": 0.0,
                                      "attachment": False, "worker": None, "preparing": 0})


def _buffer(client, chat_id, text, kind="text"):
    st = _state(chat_id)
    st["parts"].append(text)
    st["voice"] = st["voice"] or kind == "voice"
    st["attachment"] = st["attachment"] or kind == "document"
    st["last"] = time.time()
    w = st["worker"]
    if w is None or w.done():
        w = asyncio.create_task(_worker(client, chat_id))
        st["worker"] = w
        _inflight.add(w)
        w.add_done_callback(_inflight.discard)


# First contact is two beats (R4 Г1/Г5): the intro leads with who I am, what
# happens here, and a low-threshold invitation — no negations. The honest limits
# («not a therapist») live in _frames() and are sent as a second message, after
# the invitation, not as a wall of caveats before the first word. Three entry
# buttons lower the threshold further and attribute the chosen door (the funnel
# signal R4 found missing — every source was "direct").
_ENTRY_CHOICES = {
    "work": "Работа и дело",
    "relations": "Отношения",
    "decision": "Решение, что не даётся",
}
_ENTRY_PROMPTS = {
    "work": "Окей, работа и дело. Что именно сейчас цепляет — конкретная "
            "ситуация, человек или ощущение в целом? Расскажи, как есть.",
    "relations": "Окей, отношения. С кем и что происходит? Можно с любого "
                 "места — я соберу картину.",
    "decision": "Окей, решение, которое не даётся. Между чем и чем выбираешь "
                "— и что мешает выбрать? Расскажи своими словами.",
}


def _entry_keyboard():
    return {"inline_keyboard": [
        [{"text": _ENTRY_CHOICES["work"], "callback_data": "entry:work"}],
        [{"text": _ENTRY_CHOICES["relations"], "callback_data": "entry:relations"}],
        [{"text": _ENTRY_CHOICES["decision"], "callback_data": "entry:decision"}],
    ]}


def _onboarding(chat_id: int) -> str:
    return (
        "Привет. Я — коуч-разборщик: помогаю распутать то, что крутится в "
        "голове, и найти, за что в этом ухватиться.\n\n"
        "Как это работает: расскажи, что происходит — словами или голосом. "
        "Можно принести тест, опросник или стенограмму разговора. Я помогу "
        "увидеть, что за этим стоит, и один конкретный шаг дальше.\n\n"
        "С чего начнём? Опиши своими словами — или нажми, что ближе:"
    )


def _frames(chat_id: int) -> str:
    base = (
        "Пара честных рамок: я не терапевт и не заменяю помощь в кризисе — "
        "если тяжело прямо сейчас, лучше к живому специалисту.\n\n"
        "🔒 Без имени и телефона, память зашифрована. Забрать — /export, "
        "стереть — /delete_my_data, подробнее — /privacy"
    )
    if PUBLIC_MODE and not _is_admin(chat_id):
        base += f"\n\nЭто публичное демо, лимит {DAILY_MESSAGE_LIMIT} сообщений в день."
    return base


# Layered privacy notice: the first screen carries one warm line, the detail
# lives here a tap away (the just-in-time pattern the ICO recommends). It exists
# because fifteen lines of caveats were scaring people off before they wrote a
# word — five of thirteen tenants never did.
#
# Written to answer what the reader is actually afraid of — who will see this,
# is someone watching, will it be sold, can I take it back — not to describe our
# processing. The one thing it must never do is trade honesty for warmth: the
# operator really does read разборы through supervise.py, so that is said, in
# the split the industry settled on (Wysa: nobody eavesdrops on you, staff check
# the AI's answers; Woebot: we review de-identified conversations). Attention is
# on the coach's mistakes, and the sentence says exactly that — a reassurance
# the practice contradicts is worse than no reassurance at all. It stays in the
# first person on purpose: «ответы анализируются» would read as a machine doing
# it, leaving the reader unaware a person may see what they wrote.
#
# Key custody is deliberately left to the README: once the notice admits a human
# may read the разбор, "the key sits on the same server" tells the reader nothing
# new about their exposure and costs a paragraph of dread.
PRIVACY_NOTICE = (
    "🔒 Что происходит с твоими словами\n\n"
    "Общение с коучем анонимно: ни имени, ни телефона, ни почты.\n\n"
    "В реальном времени тебя не читает никто. Ответы коуча я анализирую на "
    "соответствие и ошибки — так он становится лучше.\n\n"
    "Ничего не продаётся и никому не передаётся. Рекламы здесь нет.\n\n"
    "Забрать и стереть — в один шаг: /export отдаст всё файлом, "
    "/delete_my_data сотрёт разговоры без копий. Останется обезличенная "
    "пометка «попал / мимо» — без текста и без связи с тобой.\n\n"
    "Хочешь, чтобы всё крутилось только у тебя, — код открыт:\n"
    "github.com/denimasterden-ui/personal-ai-coach"
)


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
        # First contact: persist the raw chat_id so the person is reachable even
        # after the in-RAM session is gone (the tenant hash is one-way).
        cid = str(chat_id)
        if cid not in _contacts:
            _contacts[cid] = {}
            _save_contacts()
        # ①: intro + value + entry buttons — the primary, low-threshold action.
        await _tg(client, "sendMessage", chat_id=chat_id,
                  text=_onboarding(chat_id), reply_markup=_entry_keyboard())
        # ②: honest frames as a second beat; carries the tour as a secondary offer.
        tour_done = _contacts[cid].get("tour") == "done"
        kw = {} if tour_done else {"reply_markup": _tour_offer_keyboard()}
        await _tg(client, "sendMessage", chat_id=chat_id, text=_frames(chat_id), **kw)
    elif cmd == "/memory":
        summary = await memory.summarize(tenant)
        await _send_text(client, chat_id,
                         summary or "Пока пусто — расскажи о себе, и я начну запоминать.")
    elif cmd == "/portrait":
        try:
            result = await _ask_portrait(client, tenant)
        except Exception as exc:
            print("[portrait] brain error:", exc, flush=True)
            await _tg(client, "sendMessage", chat_id=chat_id,
                      text="Не удалось собрать портрет прямо сейчас. Попробуй чуть позже.")
            return
        if not result.get("ready"):
            await _tg(client, "sendMessage", chat_id=chat_id,
                      text=result.get("reason") or "Для портрета пока ещё рано.")
        else:
            portrait = result.get("text") or ""
            await _send_text(client, chat_id, portrait)
            await _send_document(client, chat_id, "aicoach-portrait.md", portrait)
    elif cmd == "/export":
        dump = await memory.export_tenant(tenant)
        if not dump:
            await _tg(client, "sendMessage", chat_id=chat_id, text="Пока нечего экспортировать.")
        else:
            await _send_document(client, chat_id, "aicoach-memory.md", dump)
    elif cmd == "/privacy":
        await _send_text(client, chat_id, PRIVACY_NOTICE)
    elif cmd == "/delete_my_data":
        # Telegram auto-links /commands in message text, so /delete_confirm below
        # renders as a single tappable confirmation — no typed argument needed.
        await _tg(client, "sendMessage", chat_id=chat_id,
                  text="Это удалит ВСЮ память о тебе безвозвратно. Сначала можешь "
                       "забрать её через /export.\n\nЧтобы подтвердить — нажми /delete_confirm")
    elif cmd == "/delete_confirm":
        ok = await memory.delete_tenant(tenant)
        # The quality layer holds the ход verbatim too — deleting only tenants/
        # left the conversation sitting in supervision.db while the bot said
        # "вся твоя память удалена". Text goes, the anonymous rating stays.
        await asyncio.to_thread(supervision.forget_tenant, tenant)
        # also drop the live in-RAM conversation in the service, else the coach
        # keeps "remembering" from session context (and may re-save) until TTL.
        try:
            await client.delete(f"{AICOACH_URL}/session/tg-{chat_id}", timeout=10)
        except Exception as exc:
            print("[delete] session reset failed:", exc, flush=True)
        if _subs.pop(str(chat_id), None) is not None:
            _save_subs()
        if _contacts.pop(str(chat_id), None) is not None:
            _save_contacts()
        _feedback_pending.discard(chat_id)
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
    {"command": "portrait", "description": "Собрать связный портрет"},
    {"command": "export", "description": "Забрать свою память файлом .md"},
    {"command": "weekly", "description": "Еженедельное напоминание вернуться к рефлексии"},
    {"command": "privacy", "description": "Что хранится и кто это видит"},
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
        if chat_id in _feedback_pending:
            _feedback_pending.discard(chat_id)
            await _capture_feedback(client, chat_id, text)
            return
        if not analytics.has_seen(_tenant_id_for(chat_id)):
            analytics.log("first_seen", _tenant_id_for(chat_id), source="")
        _buffer(client, chat_id, text, kind="voice")
    elif "text" in msg:
        text = msg["text"]
        if text.startswith("/"):
            await _command(client, chat_id, text)
            return
        if chat_id in _feedback_pending:
            _feedback_pending.discard(chat_id)
            await _capture_feedback(client, chat_id, text)
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
        _buffer(client, chat_id, f"{head}]\n\n{preview}", kind="document")
    else:
        await _tg(client, "sendMessage", chat_id=chat_id,
                  text="Я понимаю текст, голосовые и PDF. Это пока не возьму.")


async def _drop_keyboard(client, chat_id, message_id):
    """Take the buttons off a message that has been used up. Best-effort: a
    failed edit costs nothing here, unlike the entry gate where it guards a log."""
    try:
        await _tg(client, "editMessageReplyMarkup", chat_id=chat_id,
                  message_id=message_id, reply_markup={"inline_keyboard": []})
    except Exception as exc:
        print(f"[tour] editMessageReplyMarkup failed: {exc}", flush=True)


async def _handle_tour_callback(client, cq_id, chat_id, message_id, action):
    """Handle a tour button press: advance to the next step or exit the tour.

    Every step is its own message, so without clearing the keyboard the chat is
    left littered with live «Дальше →» buttons from steps already passed — and
    the offer on the frames message stays tappable after the tour is over."""
    cid = str(chat_id)
    await _drop_keyboard(client, chat_id, message_id)
    if action == "exit":
        _contacts.setdefault(cid, {})["tour"] = "done"
        _save_contacts()
        print(f"[tour] chat={chat_id} done", flush=True)
        await _tg(client, "answerCallbackQuery", callback_query_id=cq_id,
                  text="Возвращаемся к разговору")
        return

    if not action.isdigit():
        await _tg(client, "answerCallbackQuery", callback_query_id=cq_id)
        return

    step = int(action)
    if step >= len(_TOUR_STEPS):
        await _tg(client, "answerCallbackQuery", callback_query_id=cq_id)
        return

    # Stale button: tour already done, silently acknowledge
    if _contacts.get(cid, {}).get("tour") == "done":
        await _tg(client, "answerCallbackQuery", callback_query_id=cq_id)
        return

    print(f"[tour] chat={chat_id} step={step}", flush=True)
    await _tg(client, "sendMessage", chat_id=chat_id,
              text=_TOUR_STEPS[step], reply_markup=_tour_nav_keyboard(step))
    await _tg(client, "answerCallbackQuery", callback_query_id=cq_id)


async def _handle_entry_callback(client, cq_id, chat_id, message_id, choice):
    """Handle an onboarding entry button: drop the keyboard first so a stale
    message can't be re-tapped into a second log, then log the chosen door
    (funnel attribution R4 lacked), send one narrowing question, acknowledge.
    If the keyboard can't be cleared the buttons stay live — so we don't log or
    prompt, just acknowledge and let the person tap again (recorded once).
    Never calls the brain — the person's next free-text turn does that."""
    try:
        await _tg(client, "editMessageReplyMarkup", chat_id=chat_id,
                  message_id=message_id, reply_markup={"inline_keyboard": []})
    except Exception as exc:
        print(f"[entry] editMessageReplyMarkup failed, not logging: {exc}", flush=True)
        await _tg(client, "answerCallbackQuery", callback_query_id=cq_id)
        return
    tenant = _tenant_id_for(chat_id)
    analytics.log("onboarding_choice", tenant, kind=choice)
    print(f"[entry] chat={chat_id} choice={choice}", flush=True)
    await _tg(client, "sendMessage", chat_id=chat_id, text=_ENTRY_PROMPTS[choice])
    await _tg(client, "answerCallbackQuery", callback_query_id=cq_id)


async def _handle_callback(client, cq):
    """Handle an inline button press (callback_query).
    Validates payload strictly — it's an external input. Saves the rating, updates
    the button display, acknowledges. Never produces a new chat message."""
    cq_id = cq["id"]
    data = cq.get("data", "")
    msg = cq.get("message")
    if not msg:
        await _tg(client, "answerCallbackQuery", callback_query_id=cq_id)
        return
    chat_id = msg["chat"]["id"]
    message_id = msg["message_id"]

    parts = data.split(":")
    if parts[0] == "tour" and len(parts) == 2:
        await _handle_tour_callback(client, cq_id, chat_id, message_id, parts[1])
        return

    if parts[0] == "entry" and len(parts) == 2 and parts[1] in _ENTRY_PROMPTS:
        await _handle_entry_callback(client, cq_id, chat_id, message_id, parts[1])
        return

    if (len(parts) != 3 or parts[0] != "rate"
            or not _VALID_TURN_ID.match(parts[1])
            or parts[2] not in _VALID_RATINGS):
        print(f"[rating] invalid callback_data: {data!r}", flush=True)
        await _tg(client, "answerCallbackQuery", callback_query_id=cq_id)
        return

    _, turn_id, value = parts
    tenant = _tenant_id_for(chat_id)
    supervision.save_rating(tenant, turn_id, value)
    print(f"[rating] chat={chat_id} turn={turn_id[:8]}… value={value}", flush=True)

    try:
        await _tg(client, "editMessageReplyMarkup",
                  chat_id=chat_id, message_id=message_id,
                  reply_markup=_rating_keyboard(turn_id, selected=value))
    except Exception as exc:
        print(f"[rating] editMessageReplyMarkup failed: {exc}", flush=True)

    await _tg(client, "answerCallbackQuery",
              callback_query_id=cq_id, text=_RATING_ACK[value])


async def main():
    offset = None
    mode = f"PUBLIC (hashed tenants, {DAILY_MESSAGE_LIMIT}/day cap)" if PUBLIC_MODE else (
        "private, whitelist=set" if ALLOWED_CHAT_ID else "private, whitelist=OPEN (reveals chat_id)")
    _load_subs()
    _load_contacts()
    analytics.init()
    print(f"AICOACH bot up. mode={mode}, weekly_subs={len(_subs)}, contacts={len(_contacts)}")

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
                cq = u.get("callback_query")
                if msg:
                    try:
                        await _handle(client, msg)
                    except Exception as exc:
                        print("handle error:", exc)
                elif cq:
                    try:
                        await _handle_callback(client, cq)
                    except Exception as exc:
                        print("callback error:", exc)

        print(f"[shutdown] draining {len(_inflight)} in-flight turn(s)…", flush=True)
        if _inflight:
            await asyncio.wait(set(_inflight), timeout=DRAIN_TIMEOUT)
        print("[shutdown] done", flush=True)


if __name__ == "__main__":
    asyncio.run(main())

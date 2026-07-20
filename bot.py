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
import time
from pathlib import Path

import httpx

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
    return n > DAILY_MESSAGE_LIMIT


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
                    print(f"[weekly] pushed to {cid}", flush=True)
                elif not r.get("ok") and r.get("error_code") == 403:
                    _subs.pop(cid, None)  # user blocked the bot — drop them
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
        files={"file": ("voice.oga", audio, "audio/ogg")},
        data={"model": GROQ_STT_MODEL, "language": "ru"},
        timeout=60,
    )
    if r.status_code != 200:
        print(f"[stt] groq error {r.status_code}: {r.text[:200]}", flush=True)
        return ""
    return (r.json().get("text") or "").strip()


async def _tg(client, method, **params):
    r = await client.post(f"{API}/{method}", json=params, timeout=30)
    return r.json()


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


async def _ask_brain(client, tenant_id, session_id, text, model=None) -> str:
    """POST /session, consume the SSE stream, return the final answer text."""
    answer = ""
    payload = {"tenant_id": tenant_id, "session_id": session_id, "message": text}
    if model:
        payload["model"] = model
    async with client.stream(
        "POST", f"{AICOACH_URL}/session", json=payload, timeout=300,
    ) as resp:
        event = None
        async for line in resp.aiter_lines():
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: ") and event == "answer":
                answer = json.loads(line[6:]).get("text", "")
    return answer or "(пустой ответ)"


# Debounce: a person often sends a long thought as several messages in a row.
# Instead of answering each fragment, buffer incoming text per chat and only run
# the brain once, DEBOUNCE_SEC after the last message — the fragments are joined
# into one turn (which also means one save_memory pass, not one per fragment).
DEBOUNCE_SEC = float(os.environ.get("DEBOUNCE_SEC", "4"))
_pending: dict[int, dict] = {}


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


async def _flush(client, chat_id):
    try:
        await asyncio.sleep(DEBOUNCE_SEC)
    except asyncio.CancelledError:
        return
    entry = _pending.pop(chat_id, None)
    if not entry or not entry["parts"]:
        return
    text = "\n\n".join(entry["parts"])
    print(f"[flush] chat={chat_id} parts={len(entry['parts'])} chars={len(text)}", flush=True)
    if len(text) > LONG_INPUT_CHARS:
        await _tg(client, "sendMessage", chat_id=chat_id,
                  text="Принял, обрабатываю — большой объём, это займёт пару минут…")
    model = ADMIN_MODEL if _is_admin(chat_id) else None
    ka = asyncio.create_task(_typing_keepalive(client, chat_id))
    try:
        answer = await _ask_brain(client, _tenant_id_for(chat_id), f"tg-{chat_id}", text, model=model)
    finally:
        ka.cancel()
    await _send_text(client, chat_id, answer)
    print(f"[answer] sent {len(answer)} chars", flush=True)


def _buffer(client, chat_id, text):
    entry = _pending.setdefault(chat_id, {"parts": [], "task": None})
    entry["parts"].append(text)
    if entry["task"]:
        entry["task"].cancel()
    entry["task"] = asyncio.create_task(_flush(client, chat_id))


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
        if arg == "confirm":
            ok = await memory.delete_tenant(tenant)
            if _subs.pop(str(chat_id), None) is not None:
                _save_subs()
            await _tg(client, "sendMessage", chat_id=chat_id,
                      text="Готово — вся твоя память удалена." if ok else "Данных и так не было.")
        else:
            await _tg(client, "sendMessage", chat_id=chat_id,
                      text="Это удалит ВСЮ память о тебе безвозвратно. Сначала можешь "
                           "забрать её через /export.\n\nЧтобы подтвердить: /delete_my_data confirm")
    elif cmd == "/weekly":
        if arg == "on":
            _subs[str(chat_id)] = time.time()
            _save_subs()
            await _tg(client, "sendMessage", chat_id=chat_id,
                      text="Включил. Через неделю напомню вернуться к рефлексии. Отключить — /weekly off")
        elif arg == "off":
            if _subs.pop(str(chat_id), None) is not None:
                _save_subs()
            await _tg(client, "sendMessage", chat_id=chat_id, text="Напоминания отключены.")
        else:
            on = str(chat_id) in _subs
            await _tg(client, "sendMessage", chat_id=chat_id,
                      text=f"Еженедельные напоминания: {'включены' if on else 'выключены'}.\n"
                           "Включить — /weekly on · Выключить — /weekly off")
    else:
        await _tg(client, "sendMessage", chat_id=chat_id,
                  text="Неизвестная команда. /start — список команд.")


async def _handle(client, msg):
    chat_id = msg["chat"]["id"]
    kind = "voice" if ("voice" in msg or "audio" in msg) else ("text" if "text" in msg else "other")
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
        _buffer(client, chat_id, text)
    elif "text" in msg:
        text = msg["text"]
        if text.startswith("/"):
            await _command(client, chat_id, text)
            return
        _buffer(client, chat_id, text)


async def main():
    offset = None
    mode = f"PUBLIC (hashed tenants, {DAILY_MESSAGE_LIMIT}/day cap)" if PUBLIC_MODE else (
        "private, whitelist=set" if ALLOWED_CHAT_ID else "private, whitelist=OPEN (reveals chat_id)")
    _load_subs()
    print(f"AICOACH bot up. mode={mode}, weekly_subs={len(_subs)}")
    async with httpx.AsyncClient() as client:
        asyncio.create_task(_weekly_loop(client))
        while True:
            try:
                upd = await _tg(client, "getUpdates", offset=offset, timeout=25)
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


if __name__ == "__main__":
    asyncio.run(main())

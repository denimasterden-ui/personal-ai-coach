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

import httpx

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
    if not PUBLIC_MODE:
        return False
    day = int(time.time() // 86400)
    d, n = _daily_count.get(chat_id, (day, 0))
    if d != day:
        d, n = day, 0
    n += 1
    _daily_count[chat_id] = (d, n)
    return n > DAILY_MESSAGE_LIMIT

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


async def _ask_brain(client, tenant_id, session_id, text) -> str:
    """POST /session, consume the SSE stream, return the final answer text."""
    answer = ""
    async with client.stream(
        "POST", f"{AICOACH_URL}/session",
        json={"tenant_id": tenant_id, "session_id": session_id, "message": text},
        timeout=180,
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
    ka = asyncio.create_task(_typing_keepalive(client, chat_id))
    try:
        answer = await _ask_brain(client, _tenant_id_for(chat_id), f"tg-{chat_id}", text)
    finally:
        ka.cancel()
    await _tg(client, "sendMessage", chat_id=chat_id, text=answer)
    print(f"[answer] sent {len(answer)} chars", flush=True)


def _buffer(client, chat_id, text):
    entry = _pending.setdefault(chat_id, {"parts": [], "task": None})
    entry["parts"].append(text)
    if entry["task"]:
        entry["task"].cancel()
    entry["task"] = asyncio.create_task(_flush(client, chat_id))


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
        if text.startswith("/start"):
            greeting = "Привет. Пиши или наговаривай, что происходит — разберём."
            if PUBLIC_MODE:
                greeting += (
                    f"\n\nЭто публичное демо. Лимит: {DAILY_MESSAGE_LIMIT} сообщений/день. "
                    "Память хранится изолированно и зашифрована на сервере, но для полной "
                    "приватности разверни свой инстанс: github.com/denimasterden-ui/personal-ai-coach"
                )
            await _tg(client, "sendMessage", chat_id=chat_id, text=greeting)
            return
        _buffer(client, chat_id, text)


async def main():
    offset = None
    mode = f"PUBLIC (hashed tenants, {DAILY_MESSAGE_LIMIT}/day cap)" if PUBLIC_MODE else (
        "private, whitelist=set" if ALLOWED_CHAT_ID else "private, whitelist=OPEN (reveals chat_id)")
    print(f"AICOACH bot up. mode={mode}")
    async with httpx.AsyncClient() as client:
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

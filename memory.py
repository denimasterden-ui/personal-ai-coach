"""File-based, tenant-scoped memory for AICOACH — .md by default, human-readable
and git-diffable (per Denis: memory lives in editable .md files). No Neo4j.

Isolation (R1) lives HERE, inside the tools, not in a pipe: every path is
resolved and asserted to stay inside tenants/<tenant_id>/, so the model's
freedom to call these tools whenever it likes can't leak across tenants.

Curator (R3): save_memory takes `supersedes` — old entries are removed, not
left to accumulate contradictory facts next to the new one.

Memory types (§6 of TZ) map to files under the tenant dir:
  self.md · patterns/<slug>.md · coach/<slug>.md · open_loops.md ·
  evidence.md · decisions/<slug>.md · sessions/<slug>.md

Encryption at rest (optional, off by default): if config.MEMORY_ENCRYPTION_KEY
is set, tenant memory files are written/read as Fernet ciphertext instead of
plaintext. Honest threat model — this protects against someone who gets a copy
of tenants/ without the server's .env (a stolen disk, a leaked backup), NOT
against the server operator (who holds the key alongside the data). True
"not even the operator can read it" needs a per-user passphrase-derived key,
which is a separate, not-yet-built mode. Leave the key unset for a private
self-hosted instance you want to stay human-editable; set it for a public/
multi-user deployment. Skills (shared, non-personal) are never encrypted.
"""

import asyncio
import logging
import re
import shutil
from datetime import datetime, timezone

import config

log = logging.getLogger("aicoach.memory")

_fernet = None
if config.MEMORY_ENCRYPTION_KEY:
    from cryptography.fernet import Fernet
    _fernet = Fernet(config.MEMORY_ENCRYPTION_KEY.encode())


def _write_text(path, text: str) -> None:
    data = text.encode("utf-8")
    path.write_bytes(_fernet.encrypt(data) if _fernet else data)


def _read_text(path) -> str:
    data = path.read_bytes()
    if _fernet:
        data = _fernet.decrypt(data)
    return data.decode("utf-8")


# Single-file types (append/replace a whole file) vs per-entry types (one file
# per slug in a subdir). Keeps the model's `type` argument small and explicit.
_SINGLE_FILE = {"self", "open_loops", "evidence"}
_PER_ENTRY = {"pattern", "coach", "decision", "session", "test", "doc", "loop"}
_SUBDIR = {"pattern": "patterns", "coach": "coach", "decision": "decisions",
           "session": "sessions", "test": "tests", "doc": "docs", "loop": "loops"}

_STOPWORDS = {"и", "в", "на", "по", "за", "с", "от", "для", "что", "как", "это", "к", "у",
              "the", "a", "of", "to", "is", "in", "it"}

# How many lines of a docs/<slug>.md document to show in recall's stub — enough
# to hint what's inside, not enough to flood context.
_DOC_PREVIEW_LINES = 5

# The always-on open-loops channel is capped by volume, not count — like Letta's
# char_limit on a memory block. ~1000 tokens for Russian (≈4 char/token); the real
# token length surfaces in /stats (T5) to calibrate this. Char count keeps a
# tokenizer out of recall's hot path.
_OPEN_LOOPS_CHAR_CAP = 4000


def _slug(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9а-яА-Я_-]+", "-", text.strip().lower()).strip("-")
    return s[:60] or "entry"


def _tenant_dir(tenant_id: str):
    """Resolve tenants/<tenant_id>/ and assert it stays under TENANTS_DIR —
    blocks '../' traversal in a hostile/buggy tenant_id (R1)."""
    root = config.TENANTS_DIR.resolve()
    d = (root / _slug(tenant_id)).resolve()
    if root not in d.parents and d != root:
        raise ValueError(f"tenant path escapes root: {tenant_id!r}")
    return d


def _resolve(tenant_id: str, mem_type: str, slug: str | None):
    d = _tenant_dir(tenant_id)
    if mem_type in _SINGLE_FILE:
        path = d / f"{mem_type}.md"
    elif mem_type in _PER_ENTRY:
        path = d / _SUBDIR[mem_type] / f"{_slug(slug or '')}.md"
    else:
        raise ValueError(f"unknown memory type: {mem_type!r}")
    path = path.resolve()
    if d not in path.parents:
        raise ValueError(f"memory path escapes tenant dir: {mem_type}/{slug}")
    return path


# ── write (with curator) ─────────────────────────────────────────────────────

def _reject_system_lines(text, where):
    """Refuse a body the model seeded with its own _source:/_updated:/_status:
    line. Those are system-written; a forged one (the model's made-up _updated:
    date) lands above the real footer and reads as the freshest truth. Costs one
    retry, buys an un-forged file."""
    for line in text.splitlines():
        s = line.strip()
        if s.startswith(("_updated:", "_source:", "_status:")):
            field = s.split(":", 1)[0]
            raise ValueError(
                f"запись отклонена: в {where} есть служебная строка '{field}:'. "
                "Поля _source:, _updated: и _status: проставляет система — "
                "повтори вызов без неё.")


def _write_sync(tenant_id, mem_type, slug, content, supersedes, mode, source, status):
    # The model used to drop its own "_updated: <made-up date>_" into the body —
    # sitting above the real footer it read as the freshest truth. _source: and
    # _updated: are system fields; refuse so the call is retried without them.
    _reject_system_lines(content, "content")
    # supersedes on a single-file type is a silent no-op: _resolve returns the very
    # path being written, so the `sp != path` guard below never unlinks anything,
    # yet the call reported success. Surface it instead of pretending to curate.
    if supersedes and mem_type in _SINGLE_FILE:
        raise ValueError(
            f"supersedes не применим к типу {mem_type}: это single-file тип (один "
            f"файл без slug'ов, снимать нечего). Чтобы заменить устаревшее — "
            "используй mode=\"replace\" или edit_memory.")
    path = _resolve(tenant_id, mem_type, slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    new = content.strip()
    # loop: a machine-readable status line the code writes (not the model), first
    # in the body. The loop's identity is its slug now, not its ordinal in a list.
    status_line = f"_status: {status}_\n\n" if (mem_type == "loop" and status) else ""
    # provenance: where this assertion came from (test result, user's words in
    # a session, coach's inference, profile seed...). Without it the model can't
    # tell a measured fact from a guess, so it's recorded next to the content.
    # None → no line, keeping the old (pre-provenance) file shape.
    provenance = f"_source: {' '.join(source.split())}_\n\n" if source else ""
    # single-file types (self/open_loops/evidence): mode="append" grows the file
    # (so a profile sent in fragments accumulates instead of overwriting itself),
    # mode="replace" rewrites it whole. per-entry types are one file per slug —
    # a save just replaces that slug's file, mode doesn't apply.
    if mem_type in _SINGLE_FILE and mode == "append" and path.exists():
        existing = _read_text(path).rstrip()
        body = f"{existing}\n\n---\n\n{new}\n\n{provenance}_updated: {stamp}_\n"
        action = "appended"
    else:
        body = f"{status_line}{new}\n\n{provenance}_updated: {stamp}_\n"
        action = "replaced" if mem_type in _SINGLE_FILE else "saved"
    _write_text(path, body)
    # curator: drop superseded per-entry files
    removed = []
    for stale in supersedes or []:
        try:
            sp = _resolve(tenant_id, mem_type, stale)
            if sp != path and sp.exists():
                sp.unlink()
                removed.append(stale)
        except ValueError:
            continue
    return {action: f"{mem_type}/{slug}" if slug else mem_type, "superseded": removed}


async def save_memory(tenant_id, mem_type, content, slug=None, supersedes=None, mode="append", source=None, status=None):
    try:
        return await asyncio.to_thread(
            _write_sync, tenant_id, mem_type, slug, content, supersedes, mode, source, status
        )
    except Exception as exc:
        log.warning("save_memory(%s/%s) failed: %s", mem_type, slug, exc)
        return {"error": f"{type(exc).__name__}: {exc}"}


# ── recall (token-overlap search over the tenant's .md files) ─────────────────

def _iter_md(d):
    if not d.exists():
        return
    for p in d.rglob("*.md"):
        if p.is_file():
            yield p


def _doc_stub(slug, text, rel):
    """A `docs/` source document can run to tens of thousands of chars — putting
    its full text in recall's output would crowd out everything else. Score is
    computed against the whole text (so the document is discoverable), but we
    return only a preview + how to fetch the whole thing: load_doc(slug)."""
    preview = text.strip().splitlines()
    preview = "\n".join(preview[:_DOC_PREVIEW_LINES]).strip()
    return {"path": rel, "content": f"[документ {slug}]\n{preview}\n\n…полную выдачу запроси: load_doc(\"{slug}\")",
            "doc": True, "slug": slug}


def _parse_updated(text):
    """ISO stamp from a memory footer. Falls back to '' so a missing stamp sorts
    oldest, not crashes. Stamps are written by the code (save_memory) in one
    isoformat, so they compare correctly as strings. Two footer shapes exist:
    `_updated: <iso>_` after a save and `_updated: <iso> (правка: …)_` after an
    edit — match the ISO characters themselves, or a corrected loop reads as
    undated and sinks to the bottom of the channel, exactly backwards."""
    m = re.search(r"_updated:\s*([0-9T:+.\-]+)", text)
    return m.group(1) if m else ""


def _open_loops(d):
    """Every open loop, freshest first, as (stamp, relpath, body). A loop's status
    is a code-written machine field (_status: open_), so 'open' is a filter, not a
    search. One definition of "open", shared by the model's always-on channel and
    the human-facing /memory view — they drifted apart once already."""
    loops_dir = d / "loops"
    if not loops_dir.is_dir():
        return []
    open_loops = []
    for p in loops_dir.glob("*.md"):
        try:
            text = _read_text(p)
        except Exception as exc:  # unreadable (wrong/missing key) — skip, don't poison recall
            log.warning("recall: unreadable %s: %s", p, exc)
            continue
        first = text.lstrip().splitlines()[0].strip() if text.strip() else ""
        if first == "_status: open_":
            open_loops.append((_parse_updated(text), str(p.relative_to(d)), text.strip()))
    open_loops.sort(key=lambda s: s[0], reverse=True)  # freshest first
    return open_loops


def _open_loops_channel(d):
    """Always-on context: every open loop, score-independent — these loops compete
    for no query-scored slot. Done/dropped loops stay query-relevant and ride the
    normal top-K below. Capped by chars, freshest first: on overflow the older loops
    collapse into a stub so the model knows the channel was truncated, not
    exhaustive (silent truncation is exactly the bug we're fixing).
    Returns (channel_entries, open_paths); open_paths lets _recall_sync skip them in
    the query pass so an open loop isn't returned twice."""
    open_loops = _open_loops(d)
    if not open_loops:
        return [], set()
    out, used, cut = [], 0, 0
    for _stamp, rel, body in open_loops:
        if used + len(body) <= _OPEN_LOOPS_CHAR_CAP:
            out.append({"path": rel, "content": body})
            used += len(body)
        else:
            cut += 1
    if cut:
        out.append({"path": "loops/", "content":
                    f"…+ ещё {cut} открытых петель (старее). Уточни конкретику или /memory."})
    return out, {rel for _s, rel, _b in open_loops}


def _recall_sync(tenant_id, query, limit):
    d = _tenant_dir(tenant_id)
    tokens = [t for t in re.findall(r"\w+", query.lower()) if len(t) >= 3 and t not in _STOPWORDS]
    # open loops are always-on context (first in output, score-independent);
    # excluded from the query pass below so they aren't returned twice.
    open_channel, open_paths = _open_loops_channel(d)
    scored = []
    for p in _iter_md(d):
        if str(p.relative_to(d)) in open_paths:
            continue
        try:
            text = _read_text(p)
        except Exception as exc:  # incl. cryptography.fernet.InvalidToken (wrong/missing key)
            log.warning("recall: unreadable %s: %s", p, exc)
            continue
        hay = text.lower()
        score = sum(hay.count(t) for t in tokens) if tokens else 0
        # self.md is always relevant context — floor its score so it surfaces
        if p.name == "self.md":
            score = max(score, 1)
        if score:
            rel = str(p.relative_to(d))
            scored.append((score, rel, text, p))
    scored.sort(key=lambda s: s[0], reverse=True)
    out = list(open_channel)
    for _, rel, text, p in scored[:limit]:
        # docs/<slug>.md: keep full text out of recall, surface a stub instead
        if p.parent.name == "docs":
            out.append(_doc_stub(p.stem, text, rel))
        else:
            out.append({"path": rel, "content": text.strip()})
    return out


async def recall(tenant_id, query, limit=5):
    try:
        return await asyncio.to_thread(_recall_sync, tenant_id, query, limit)
    except Exception as exc:
        log.warning("recall(%r) failed: %s", query, exc)
        return []


def _has_derived_records_sync(tenant_id):
    """Whether the coach has already extracted something from this person's
    conversation. Uploaded source documents deliberately do not count: the bot
    stores them before the first analysis."""
    d = _tenant_dir(tenant_id)
    return any(p.parent.name != "docs" for p in _iter_md(d))


async def has_derived_records(tenant_id):
    try:
        return await asyncio.to_thread(_has_derived_records_sync, tenant_id)
    except Exception as exc:
        log.warning("has_derived_records(%r) failed: %s", tenant_id, exc)
        return False


def _load_doc_sync(tenant_id, slug):
    """Fetch a source document whole by its slug. recall only shows a preview;
    the model gets the rest on demand via load_doc."""
    path = _resolve(tenant_id, "doc", slug)
    if not path.exists():
        return None
    return _read_text(path).strip()


async def load_doc(tenant_id, slug):
    try:
        return await asyncio.to_thread(_load_doc_sync, tenant_id, slug)
    except Exception as exc:
        log.warning("load_doc(%s) failed: %s", slug, exc)
        return None


def _edit_sync(tenant_id, mem_type, slug, old, new, source):
    """Replace one exact fragment in a memory file. Single-file memory could only
    append or be rewritten whole, so correcting one wrong claim meant either
    leaving it above the correction (the model then sees both) or regenerating
    tens of KB of profile. Uniqueness is required: an ambiguous match would edit
    a place the model didn't mean."""
    _reject_system_lines(new, "new_string")
    path = _resolve(tenant_id, mem_type, slug)
    if not path.exists():
        return {"error": f"нет такой записи: {mem_type}" + (f"/{slug}" if slug else "")}
    text = _read_text(path)
    hits = text.count(old)
    if hits == 0:
        return {"error": "фрагмент не найден дословно — перечитай запись через recall "
                         "и повтори с точной цитатой"}
    if hits > 1:
        return {"error": f"фрагмент встречается {hits} раз — возьми более длинную "
                         "цитату, чтобы правка попала в нужное место"}
    body = text.replace(old, new)
    # keep one footer: the newest edit and where its correction came from
    body = re.sub(r"\n*_updated: [^\n]*_\s*$", "", body).rstrip()
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    body += f"\n\n_updated: {stamp} (правка: {' '.join(source.split())})_\n"
    _write_text(path, body)
    return {"edited": f"{mem_type}/{slug}" if slug else mem_type}


async def edit_memory(tenant_id, mem_type, old, new, slug=None, source=None):
    try:
        return await asyncio.to_thread(_edit_sync, tenant_id, mem_type, slug, old, new, source)
    except Exception as exc:
        log.warning("edit_memory(%s/%s) failed: %s", mem_type, slug, exc)
        return {"error": f"{type(exc).__name__}: {exc}"}


# ── user-facing memory ops (bot commands: /memory, /export, /delete_my_data) ──

def _one(d, name):
    p = d / name
    if not p.exists():
        return ""
    try:
        text = _read_text(p)
    except Exception as exc:
        log.warning("read %s failed: %s", p, exc)
        return ""
    return _strip_footer(text)


# How many entries of one kind /memory shows in full before collapsing the rest
# into a "…и ещё N". Enough to recognise what the coach is holding, few enough
# that a long-running profile doesn't answer /memory with a wall of text.
_SUMMARY_ENTRIES = 5


def _strip_footer(text):
    """Drop the internal machine fields — `_updated:` footer, `_status: open_`
    header — for the human-facing /memory view."""
    text = re.sub(r"\n*_updated: [^\n]*_\n*", "\n\n", text)
    return re.sub(r"^_status: \w+_\n*", "", text.lstrip()).strip()


def _recent_entries(d, sub):
    """Entries of one per-entry type, freshest first, footers stripped."""
    sd = d / sub
    if not sd.is_dir():
        return []
    out = []
    for p in sd.glob("*.md"):
        try:
            text = _read_text(p)
        except Exception as exc:
            log.warning("summarize: unreadable %s: %s", p, exc)
            continue
        out.append((_parse_updated(text), _strip_footer(text)))
    out.sort(key=lambda e: e[0], reverse=True)
    return [body for _stamp, body in out if body]


def _section(title, bodies):
    """A capped section: the freshest entries in full, the rest as a count."""
    if not bodies:
        return []
    shown = bodies[:_SUMMARY_ENTRIES]
    block = f"## {title}\n\n" + "\n\n---\n\n".join(shown)
    extra = len(bodies) - len(shown)
    if extra:
        block += f"\n\n_…и ещё {extra} — целиком в /export._"
    return [block]


def _summarize_sync(tenant_id):
    """'/memory': a human-facing view of what the coach currently holds — the
    stable self, the open threads, what it has noticed, and counts of the rest.
    Not the raw file dump (that's /export).

    Open threads and patterns are shown as text, not tallied: «1 паттернов» is a
    fact about our filesystem, while the question behind /memory is «что ты про
    меня понял». Counts stay for the bulk types (sessions, docs), where the
    number is the honest answer."""
    d = _tenant_dir(tenant_id)
    parts = []
    s = _one(d, "self.md")
    if s:
        parts.append(s)
    # loops/<slug>.md is the live channel; open_loops.md is the legacy single
    # file, deprecated for writing but still on disk for long-standing tenants.
    loops = [_strip_footer(body) for _stamp, _rel, body in _open_loops(d)]
    legacy = _one(d, "open_loops.md")
    if legacy:
        loops.append(legacy)
    parts += _section("Открытые темы", loops)
    parts += _section("Что я заметил", _recent_entries(d, "patterns"))
    counts = []
    for sub, label in (("decisions", "решений"), ("sessions", "сессий"),
                       ("tests", "результатов тестов"), ("docs", "документов")):
        sd = d / sub
        n = sum(1 for _ in sd.glob("*.md")) if sd.is_dir() else 0
        if n:
            counts.append(f"{n} {label}")
    if counts:
        lead = "Также сохранено" if parts else "Сохранено"
        parts.append(f"_{lead}: " + ", ".join(counts) + "._")
    if not parts:
        return ("Пока я почти ничего о тебе не записал — профиль наполняется по мере "
                "разговора. Расскажи, что происходит, и я начну складывать картину.")
    return "\n\n".join(parts)


async def summarize(tenant_id):
    return await asyncio.to_thread(_summarize_sync, tenant_id)


def _export_sync(tenant_id):
    """'/export': all of a tenant's memory .md (decrypted) as one portable
    markdown document — the user's own data, theirs to take."""
    d = _tenant_dir(tenant_id)
    chunks = []
    for p in sorted(_iter_md(d)):
        try:
            text = _read_text(p)
        except Exception as exc:
            log.warning("export: unreadable %s: %s", p, exc)
            continue
        chunks.append(f"# {p.relative_to(d)}\n\n{text.strip()}\n")
    return "\n\n".join(chunks)


async def export_tenant(tenant_id):
    return await asyncio.to_thread(_export_sync, tenant_id)


def _delete_sync(tenant_id):
    d = _tenant_dir(tenant_id)
    if d.exists():
        shutil.rmtree(d)
        return True
    return False


async def delete_tenant(tenant_id):
    """'/delete_my_data': remove the tenant's entire memory dir. Irreversible."""
    try:
        return await asyncio.to_thread(_delete_sync, tenant_id)
    except Exception as exc:
        log.warning("delete_tenant(%s) failed: %s", tenant_id, exc)
        return False


# ── skills (shared optics/methods; reused from ops-agent SKILL.md pattern) ────

def _parse_skill(path):
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {"title": path.parent.name, "description": "", "content": text.strip()}
    meta = {}
    for line in parts[1].strip().splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip()
    meta["content"] = parts[2].strip()
    return meta


def _catalog_sync():
    out = []
    if not config.SKILLS_DIR.exists():
        return out
    for skill_dir in sorted(config.SKILLS_DIR.iterdir()):
        f = skill_dir / "SKILL.md"
        if f.is_file():
            m = _parse_skill(f)
            if m:
                out.append({"title": m.get("title", skill_dir.name),
                            "description": m.get("description", "")})
    return out


async def load_skills_catalog():
    return await asyncio.to_thread(_catalog_sync)


def _load_skill_sync(title):
    if not config.SKILLS_DIR.exists():
        return None
    for skill_dir in config.SKILLS_DIR.iterdir():
        f = skill_dir / "SKILL.md"
        if f.is_file():
            m = _parse_skill(f)
            if m and m.get("title", "").strip().lower() == title.strip().lower():
                return m["content"]
    return None


async def load_skill(title):
    return await asyncio.to_thread(_load_skill_sync, title)

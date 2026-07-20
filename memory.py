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
_PER_ENTRY = {"pattern", "coach", "decision", "session"}
_SUBDIR = {"pattern": "patterns", "coach": "coach", "decision": "decisions", "session": "sessions"}

_STOPWORDS = {"и", "в", "на", "по", "за", "с", "от", "для", "что", "как", "это", "к", "у",
              "the", "a", "of", "to", "is", "in", "it"}


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

def _write_sync(tenant_id, mem_type, slug, content, supersedes, mode):
    path = _resolve(tenant_id, mem_type, slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    new = content.strip()
    # single-file types (self/open_loops/evidence): mode="append" grows the file
    # (so a profile sent in fragments accumulates instead of overwriting itself),
    # mode="replace" rewrites it whole. per-entry types are one file per slug —
    # a save just replaces that slug's file, mode doesn't apply.
    if mem_type in _SINGLE_FILE and mode == "append" and path.exists():
        existing = _read_text(path).rstrip()
        body = f"{existing}\n\n---\n\n{new}\n\n_updated: {stamp}_\n"
        action = "appended"
    else:
        body = f"{new}\n\n_updated: {stamp}_\n"
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


async def save_memory(tenant_id, mem_type, content, slug=None, supersedes=None, mode="append"):
    try:
        return await asyncio.to_thread(
            _write_sync, tenant_id, mem_type, slug, content, supersedes, mode
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


def _recall_sync(tenant_id, query, limit):
    d = _tenant_dir(tenant_id)
    tokens = [t for t in re.findall(r"\w+", query.lower()) if len(t) >= 3 and t not in _STOPWORDS]
    scored = []
    for p in _iter_md(d):
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
            scored.append((score, rel, text.strip()))
    scored.sort(key=lambda s: s[0], reverse=True)
    return [{"path": rel, "content": text} for _, rel, text in scored[:limit]]


async def recall(tenant_id, query, limit=5):
    try:
        return await asyncio.to_thread(_recall_sync, tenant_id, query, limit)
    except Exception as exc:
        log.warning("recall(%r) failed: %s", query, exc)
        return []


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
    # drop the internal "_updated: ..." footers for the human-facing /memory view
    return re.sub(r"\n*_updated: [^\n]*_\n*", "\n\n", text).strip()


def _summarize_sync(tenant_id):
    """'/memory': a human-facing view of what the coach currently holds — the
    stable self, open threads, and counts of the rest. Not the raw file dump
    (that's /export)."""
    d = _tenant_dir(tenant_id)
    parts = []
    s = _one(d, "self.md")
    if s:
        parts.append(s)
    loops = _one(d, "open_loops.md")
    if loops:
        parts.append("## Открытые темы\n\n" + loops)
    counts = []
    for sub, label in (("patterns", "паттернов"), ("decisions", "решений"), ("sessions", "сессий")):
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

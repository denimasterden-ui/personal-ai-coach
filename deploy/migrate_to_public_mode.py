"""One-off: migrate a single-tenant private-mode instance's plaintext profile
into a PUBLIC_MODE instance's hashed, (optionally) encrypted tenant — useful
when merging a private single-user deployment into a shared/public one instead
of running two separate instances.

Run ON THE SERVER with the target instance's venv + its .env sourced (so
config.MEMORY_ENCRYPTION_KEY/TENANT_SALT match that instance), with the old
plaintext tenant tree available for reading:

  cd /path/to/target/aicoach-service
  set -a; . ./.env; set +a
  OLD_TENANT_DIR=/path/to/old/tenants/<old-tenant-id> \
  OPERATOR_CHAT_ID=<telegram-chat-id> \
  .venv/bin/python deploy/migrate_to_public_mode.py
"""

import asyncio
import hashlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
import memory  # noqa: E402

OLD_TENANT_DIR = Path(os.environ["OLD_TENANT_DIR"])
OPERATOR_CHAT_ID = os.environ["OPERATOR_CHAT_ID"]
TENANT_SALT = os.environ["TENANT_SALT"]

TYPE_BY_FILENAME = {"self.md": "self", "open_loops.md": "open_loops", "evidence.md": "evidence"}
SUBDIR_TYPE = {"patterns": "pattern", "coach": "coach", "decisions": "decision", "sessions": "session"}


async def main():
    new_tenant = hashlib.sha256(f"{TENANT_SALT}:{OPERATOR_CHAT_ID}".encode()).hexdigest()[:24]
    print(f"migrating {OLD_TENANT_DIR} -> tenants/{new_tenant} (encrypted={bool(config.MEMORY_ENCRYPTION_KEY)})")

    migrated = 0
    for f in sorted(OLD_TENANT_DIR.glob("*.md")):
        mem_type = TYPE_BY_FILENAME.get(f.name)
        if not mem_type:
            continue
        content = f.read_text(encoding="utf-8")
        r = await memory.save_memory(new_tenant, mem_type, content, mode="replace")
        print(f"  {f.name} -> {mem_type}: {r}")
        migrated += 1

    for subdir, mem_type in SUBDIR_TYPE.items():
        d = OLD_TENANT_DIR / subdir
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.md")):
            slug = f.stem
            content = f.read_text(encoding="utf-8")
            r = await memory.save_memory(new_tenant, mem_type, content, slug=slug)
            print(f"  {subdir}/{f.name} -> {mem_type}/{slug}: {r}")
            migrated += 1

    print(f"done: {migrated} files migrated into tenants/{new_tenant}")


if __name__ == "__main__":
    asyncio.run(main())

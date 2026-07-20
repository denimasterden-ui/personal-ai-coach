"""AICOACH config. Local dev talks to OpenRouter directly (Mac can't reach the
server's internal LiteLLM on localhost:4000); on deploy, point LLM_BASE_URL at
the internal LiteLLM instead — same OpenAI-compatible interface, no code change.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).parent

# OpenAI-compatible endpoint. Local dev default = OpenRouter; deploy overrides
# with the internal LiteLLM URL via env.
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1")
LLM_API_KEY = os.environ["LLM_API_KEY"]
# CORE model (O3: Claude). OpenRouter model id — override via env.
LLM_MODEL = os.environ.get("LLM_MODEL", "anthropic/claude-sonnet-4.5")

HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8091"))

# Per-tenant memory root: tenants/<tenant_id>/... . All memory reads/writes are
# forced inside a tenant's dir (R1 isolation), enforced in memory.py.
TENANTS_DIR = Path(os.environ.get("TENANTS_DIR", str(BASE_DIR / "tenants")))

# Shared skills (schools/methods) — NOT tenant-scoped: the coaching optics are
# common. Tenant-specific coach adaptations live under the tenant dir instead.
SKILLS_DIR = Path(os.environ.get("SKILLS_DIR", str(BASE_DIR / "skills")))

# Optional Fernet key (base64, `Fernet.generate_key()`) — encrypts tenant memory
# at rest when set. See memory.py module docstring for the honest threat model
# (protects a leaked backup/disk, not the operator holding this same key).
# Empty = plaintext .md, as before (default — keeps a private instance editable).
MEMORY_ENCRYPTION_KEY = os.environ.get("MEMORY_ENCRYPTION_KEY", "")

# Reasoning effort for reasoning-capable models (OpenRouter `reasoning.effort`:
# "low"/"medium"/"high"). Empty = don't send it (model's own default). Heavy
# reasoners (e.g. kimi-k3) otherwise burn the whole max_tokens budget on hidden
# reasoning — starving the answer and blowing latency — so cap it here.
REASONING_EFFORT = os.environ.get("REASONING_EFFORT", "")

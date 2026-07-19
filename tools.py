"""Memory tools the brain calls itself (no data/sandbox tools — coaching needs
hands to its own memory, not to databases). The model decides when to recall
and when to save; isolation/curator are enforced inside memory.py, not here.
"""

import memory

MEM_TYPES = ["self", "pattern", "coach", "open_loops", "evidence", "decision", "session"]

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": (
                "Найди в памяти пользователя релевантное текущему разговору: профиль (self), "
                "паттерны, открытые темы (open_loops), доказательства изменений (evidence), "
                "прошлые сессии, решения. Зови в начале разбора и когда нужен контекст о человеке."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "О чём вспомнить (тема, паттерн, имя, ситуация)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": (
                "Запиши/обнови память о пользователе, чтобы профиль рос между сессиями. "
                "Для single-file типов (self/open_loops/evidence) выбирай mode: 'append' — "
                "дополнить существующее (по умолчанию; важно, если профиль приходит частями — "
                "не затирай уже записанное), 'replace' — переписать файл целиком (когда факт "
                "устарел/изменился). Для pattern/coach/decision/session указывай slug; supersedes — "
                "slug'и устаревших записей. Помечай гипотезы как гипотезы, не как факты."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": MEM_TYPES,
                             "description": "self/open_loops/evidence — один файл; pattern/coach/decision/session — по slug"},
                    "content": {"type": "string", "description": "Содержимое в markdown"},
                    "mode": {"type": "string", "enum": ["append", "replace"],
                             "description": "Только для single-file типов: дополнить или переписать целиком. По умолчанию append."},
                    "slug": {"type": "string", "description": "Короткий идентификатор для pattern/coach/decision/session"},
                    "supersedes": {"type": "array", "items": {"type": "string"},
                                   "description": "slug'и устаревших записей того же типа, которые заменяет эта"},
                },
                "required": ["type", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_skill",
            "description": "Подгрузи полный текст оптики/метода из каталога скиллов по title, когда он релевантен разбору.",
            "parameters": {
                "type": "object",
                "properties": {"title": {"type": "string"}},
                "required": ["title"],
            },
        },
    },
]


async def dispatch(name, args, tenant_id):
    if name == "recall":
        return await memory.recall(tenant_id, args["query"])
    if name == "save_memory":
        return await memory.save_memory(
            tenant_id, args["type"], args["content"],
            slug=args.get("slug"), supersedes=args.get("supersedes"),
            mode=args.get("mode", "append"),
        )
    if name == "load_skill":
        body = await memory.load_skill(args["title"])
        return {"title": args["title"], "content": body} if body else {"error": "скилл не найден"}
    return {"error": f"unknown tool: {name}"}

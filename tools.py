"""Memory tools the brain calls itself (no data/sandbox tools — coaching needs
hands to its own memory, not to databases). The model decides when to recall
and when to save; isolation/curator are enforced inside memory.py, not here.
"""

import memory

MEM_TYPES = ["self", "pattern", "coach", "open_loops", "evidence", "decision", "session",
             "test", "doc"]

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
                             "description": "self/open_loops/evidence — один файл; "
                             "pattern/coach/decision/session/test/doc — по slug. "
                             "test — результат психометрического теста (инструмент, дата, расклад); "
                             "doc — распознанный текст присланного отчёта целиком (recall отдаёт только "
                             "превью, полный текст — через load_doc(slug))."},
                    "content": {"type": "string", "description": "Содержимое в markdown"},
                    "mode": {"type": "string", "enum": ["append", "replace"],
                             "description": "Только для single-file типов: дополнить или переписать целиком. По умолчанию append."},
                    "slug": {"type": "string", "description": "Короткий идентификатор для pattern/coach/decision/session"},
                    "supersedes": {"type": "array", "items": {"type": "string"},
                                   "description": "slug'и устаревших записей того же типа, которые заменяет эта"},
                    "source": {"type": "string",
                               "description": "Откуда этот факт: результат теста (назови какой), слова человека в "
                               "сессии, твой вывод/гипотеза, профиль при онбординге и т.п. Без источника факт "
                               "и гипотеза лежат неотличимо — указывай, чтобы не ссылаться на проверенное то, "
                               "что им не является."},
                },
                "required": ["type", "content", "source"],
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
    {
        "type": "function",
        "function": {
            "name": "load_doc",
            "description": (
                "Выдай целиком исходный документ по его slug — распознанный текст отчёта, "
                "который recall показал только превью. Зови, когда нужно процитировать "
                "конкретное место из отчёта (тест Hogan, DISC и т.п.)."
            ),
            "parameters": {
                "type": "object",
                "properties": {"slug": {"type": "string", "description": "slug документа из выдачи recall"}},
                "required": ["slug"],
            },
        },
    },
]


async def dispatch(name, args, tenant_id):
    if name == "recall":
        return await memory.recall(tenant_id, args["query"])
    if name == "save_memory":
        # Gate, not prose: a description in the schema is advisory and the model
        # skips it under load — the same lesson the invest project paid for
        # (rules in the prompt fixed nothing; the code gate did). Refusing the
        # write costs one retry and buys a memory where every assertion declares
        # its category. "моя гипотеза" is a perfectly good answer — silence isn't.
        if not (args.get("source") or "").strip():
            return {"error": "Запись отклонена: не указан source. Назови, откуда этот "
                             "факт — результат теста (какого именно), слова человека в "
                             "сессии, или твой собственный вывод/гипотеза. Повтори вызов."}
        # _resolve falls back to a shared "entry" filename when slug is missing,
        # so a second sourceless save silently overwrites the first and still
        # reports success. Harmless for a pattern; for a test result or an
        # uploaded report it destroys the very evidence a profile claim rests on.
        if args["type"] in ("test", "doc") and not (args.get("slug") or "").strip():
            return {"error": f"Запись отклонена: для типа {args['type']} нужен slug — "
                             "без него запись затрёт предыдущую. Дай короткий "
                             "идентификатор (например clifton-2021, hogan-2024)."}
        return await memory.save_memory(
            tenant_id, args["type"], args["content"],
            slug=args.get("slug"), supersedes=args.get("supersedes"),
            mode=args.get("mode", "append"), source=args.get("source"),
        )
    if name == "load_skill":
        body = await memory.load_skill(args["title"])
        return {"title": args["title"], "content": body} if body else {"error": "скилл не найден"}
    if name == "load_doc":
        body = await memory.load_doc(tenant_id, args["slug"])
        return {"slug": args["slug"], "content": body} if body is not None else {"error": "документ не найден"}
    return {"error": f"unknown tool: {name}"}

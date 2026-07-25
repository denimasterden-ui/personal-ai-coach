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
                             "test — результат психометрического теста (инструмент, дата, расклад) "
                             "и твой разбор этого результата — сюда, целиком попадает в recall; "
                             "doc — сырой текст присланного файла, его пишет бот, тебе писать туда "
                             "нельзя (recall отдаёт по нему только превью, целиком — load_doc(slug))."},
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
            "name": "edit_memory",
            "description": (
                "Точечно исправь одно место в уже записанной памяти: old_string "
                "заменяется на new_string. Бери для исправления устаревшего или "
                "неверного утверждения — вместо того чтобы дописывать поправку "
                "(тогда неверное останется выше и ты будешь видеть оба) или "
                "переписывать файл целиком (потеряешь остальное). Фрагмент должен "
                "встречаться ровно один раз — цитируй дословно, с запасом контекста."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": MEM_TYPES,
                             "description": "тип записи, которую правишь"},
                    "old_string": {"type": "string", "description": "дословный фрагмент, который заменяем"},
                    "new_string": {"type": "string", "description": "чем заменяем"},
                    "slug": {"type": "string", "description": "для типов по slug"},
                    "source": {"type": "string",
                               "description": "откуда взято исправление — результат теста, "
                               "слова человека, твой вывод"},
                },
                "required": ["type", "old_string", "new_string", "source"],
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
        # _resolve falls back to a shared "entry" filename when slug is missing, so
        # a second slugless save silently overwrites the first and still reports
        # success. Never fired in production (no entry.md exists), but it is the
        # same quiet-data-loss path either way. Read from memory._PER_ENTRY so a
        # new per-entry type is covered the day it is added, not the day it bites.
        # doc is a transport artefact: the bot writes it when a file arrives, and
        # recall deliberately collapses it to a stub. A разбор filed there becomes
        # a 249-char preview instead of a finding — which is exactly what happened
        # on the first live upload. The model's structured result belongs in test.
        if args["type"] == "doc":
            return {"error": "Тип doc — это сырой текст присланного файла, его "
                             "сохраняет бот. Свой разбор результата теста сохраняй "
                             "типом test: он попадает в recall целиком, а doc — "
                             "только превью."}
        if args["type"] in memory._PER_ENTRY and not (args.get("slug") or "").strip():
            return {"error": f"Запись отклонена: для типа {args['type']} нужен slug — "
                             "без него запись затрёт предыдущую. Дай короткий "
                             "идентификатор (например clifton-2021, uhod-iz-pizzy)."}
        return await memory.save_memory(
            tenant_id, args["type"], args["content"],
            slug=args.get("slug"), supersedes=args.get("supersedes"),
            mode=args.get("mode", "append"), source=args.get("source"),
        )
    if name == "edit_memory":
        if not (args.get("source") or "").strip():
            return {"error": "Правка отклонена: не указан source — назови, откуда "
                             "взято исправление."}
        if args["type"] == "doc":
            return {"error": "Тип doc — сырой текст присланного файла, его не правят: "
                             "это исходник, на который ссылаются другие записи."}
        return await memory.edit_memory(
            tenant_id, args["type"], args["old_string"], args["new_string"],
            slug=args.get("slug"), source=args.get("source"))
    if name == "load_skill":
        body = await memory.load_skill(args["title"])
        return {"title": args["title"], "content": body} if body else {"error": "скилл не найден"}
    if name == "load_doc":
        body = await memory.load_doc(tenant_id, args["slug"])
        return {"slug": args["slug"], "content": body} if body is not None else {"error": "документ не найден"}
    return {"error": f"unknown tool: {name}"}

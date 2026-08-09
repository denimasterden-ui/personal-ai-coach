"""Memory extraction pass — fire-and-forget after the coaching turn.

The coach no longer writes memory during the turn (aicoach-dev#10, ADR 0005).
What to remember is decided by this separate pass, which runs after the answer
is delivered, in the background, invisible to the human.

Pattern: like ``supervision.capture_async`` — fire-and-forget, never blocks
the coaching flow, never raises into the coaching flow. A crashed pass costs
a lost write, not a lost answer.

The pass has its own tool set (``tools.PASS_TOOLS``): ``recall`` for dedup,
``save_memory`` and ``edit_memory`` for writes. The coach's write tools are
gone — not "not recommended", but absent from the schemas.

Budget: 3 steps — enough for recall (dedup), write, and retry after a gate
rejection (e.g., missing source). The pass is a loop, not a single call:
the gate refuses a write and demands a retry, otherwise the refused write
is silently lost.

Rules for what and how to write live here (in ``_PASS_PROMPT``) — the single
source of truth. They were removed from the coach's system prompt and the
playbooks to avoid two versions of truth (the PROD-1 disease).
"""

import asyncio
import json
import logging

import tools

log = logging.getLogger("aicoach.recollect")

PASS_MAX_STEPS = 3  # сверка, запись, повтор после отказа гейта

# Strong reference to in-flight tasks so they aren't GC'd mid-flight.
# Done callback discards the task when finished (same pattern as supervision).
_tasks: set[asyncio.Task] = set()


_PASS_PROMPT = """\
Ты — модуль извлечения памяти из завершённого хода коучинга. Тебе даны сообщение \
человека и ответ коуча. Твоя задача — определить, что из этого разговора стоит \
запомнить, и записать это в память пользователя.

## Что запоминать

1. **Факты о пользователе** (тип self): имя, роль, компания, контекст, предпочтения, \
   ценности — то, что устойчиво между сессиями. Для self используй mode='append' \
   по умолчанию (профиль приходит частями — не затирай уже записанное), \
   mode='replace' только когда факт устарел и его надо переписать целиком.
2. **Паттерны** (тип pattern, нужен slug): повторяющиеся ситуации, реакции, \
   защитные стратегии, внутренние конфликты.
3. **Решения и обязательства** (тип decision, нужен slug): что пользователь решил, \
   что обещал себе или другим.
4. **Открытые петли** (тип loop, нужен slug и status): незавершённые дела с \
   проверяемым статусом — сделать/позвонить/выступить/проверить. status='open' \
   для новых, 'done' для выполненных, 'dropped' для брошенных.
5. **Доказательства изменений** (тип evidence): конкретные случаи, когда человек \
   повёл себя по-новому, заметил сдвиг, получил обратную связь.
6. **Результаты тестов** (тип test, нужен slug): психометрические результаты \
   (инструмент, дата, расклад) и разбор этих результатов — целиком.

## Явная просьба записать

Если человек прямо попросил: «запиши это», «сохрани», «запомни», «давай запишем» — \
ты **обязан** записать. Просьба лежит в тексте сообщения, которое ты и так получил. \
Записывай дословно или близко к тексту, тип выбирай по содержанию (факт → self, \
обязательство → loop, решение → decision).

## Что НЕ запоминать

- Разовые вопросы без контекста.
- Общие рассуждения коуча (они в ответе, не в памяти).
- То, что уже есть в памяти (сверяйся через recall).

## Как работать

1. **Сначала recall** — вызови с ключевыми словами из разговора, чтобы проверить, \
   что уже записано. Не плодь дубли.
2. **Потом save_memory / edit_memory** — если есть что-то новое и важное.
3. **Повтор после отказа** — если save_memory вернул ошибку (например, «не указан \
   source» или «нужен slug»), повтори вызов с правильными параметрами. У тебя \
   три шага, один отказ не должен стоить записи.

## Правила записи

- **source обязателен**: всегда указывай, откуда факт — «слова человека в сессии», \
  «результат теста (назови какой)», «мой вывод/гипотеза». Без источника факт и \
  гипотеза лежат неотличимо.
- **НЕ пиши в тип doc**: это сырой текст присланного файла, его пишет бот. \
  Свой разбор результата теста сохраняй типом test.
- **НЕ пиши в тип open_loops**: он устарел и закрыт для записи. Новую открытую \
  петлю записывай типом loop (с обязательным status: open/done/dropped).
- **Для per-entry типов (pattern/coach/decision/session/test/doc/loop) нужен slug**: \
  короткий идентификатор латиницей. Без slug запись затрёт предыдущую.
- **Для loop нужен status**: один из open (открыта), done (выполнена), dropped (брошена).
- **supersedes**: если уточняешь старую запись, укажи slug'и устаревших записей \
  того же типа — они будут удалены.
- **НЕ копируй `_source:`/`_updated:`/`_status:` из того, что вернул recall**: это \
  служебные строки, их всегда проставляет система при записи. Если ты видишь их в \
  найденной записи и включишь дословно в свой content — запись отклонится, а ты \
  потратишь на это шаг бюджета.

## Дедупликация

Перед записью проверь через recall, что этого ещё нет. Не записывай то, что уже \
есть, — только если факт изменился (тогда edit_memory или save_memory с supersedes).

## Завершение

Когда записал всё, что стоит запомнить (или если записывать нечего) — закончи ход \
без tool-вызовов. Не отчитывайся о том, что записал: человек этого не видит, \
память он откроет через /memory.
"""


def after_turn_async(tenant_id, user_text, answer, *, client, model):
    """Schedule memory extraction after the turn completes.

    Fire-and-forget: never blocks, never raises into the coaching flow. The
    pass runs in background; a crashed pass costs a lost write, not a lost
    answer.

    Args:
        tenant_id: Tenant identifier for memory storage.
        user_text: What the human said in the turn.
        answer: What the coach answered.
        client: Model client (OpenAI-compatible). Injected, not hardcoded —
            tests pass a fake, no network.
        model: Model id to use for the pass.
    """
    task = asyncio.create_task(
        _run_pass(tenant_id, user_text, answer, client, model)
    )
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)


async def _run_pass(tenant_id, user_text, answer, client, model):
    """Run the extraction pass. Catches all exceptions — a crashed pass is
    logged, not raised. The turn is already complete; this is background."""
    try:
        summary = await _run_pass_inner(tenant_id, user_text, answer, client, model)
        crashed = False
        error = None
    except Exception as exc:
        log.warning("recollect pass failed for %s: %s", tenant_id, exc, exc_info=True)
        summary = {
            "tool_names": [],
            "write_count": 0,
            "write_attempts": 0,
            "recall_count": 0,
        }
        crashed = True
        error = str(exc)

    # Trace the pass — fire-and-forget via supervision (aicoach-dev#11)
    import supervision  # local to avoid cycle at module load
    supervision.log_pass_async(
        tenant_id,
        tool_names=summary["tool_names"],
        write_count=summary["write_count"],
        write_attempts=summary["write_attempts"],
        recall_count=summary["recall_count"],
        crashed=crashed,
        error=error,
        user_text=user_text,
        answer=answer,
    )
    # pass_stats().no_write is an aggregate — queue the specific turn too, so
    # a silent extraction failure surfaces a case, not just a weekly count.
    if not crashed and summary["write_count"] == 0:
        supervision.capture_no_write_async(
            tenant_id, user_text, answer, write_attempts=summary["write_attempts"])


async def _run_pass_inner(tenant_id, user_text, answer, client, model):
    """Inner pass loop. Separated so the outer wrapper can catch all exceptions.

    Returns a summary dict with tool_names, write_count, write_attempts,
    recall_count — used by supervision.log_pass to trace what the pass did.
    """
    messages = [
        {"role": "system", "content": _PASS_PROMPT},
        {"role": "user", "content": f"Сообщение человека:\n{user_text}\n\n---\n\nОтвет коуча:\n{answer}"},
    ]

    tool_names = set()
    write_count = 0
    write_attempts = 0
    recall_count = 0

    for step in range(PASS_MAX_STEPS):
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools.PASS_TOOLS,
        )
        message = response.choices[0].message

        if not message.tool_calls:
            # Pass decided nothing to write, or finished writing.
            break

        messages.append(message.model_dump(exclude_none=True))

        # Execute tool calls and append results.
        for call in message.tool_calls:
            name = call.function.name
            tool_names.add(name)
            try:
                args = json.loads(call.function.arguments or "{}")
                result = await tools.dispatch(name, args, tenant_id)

                # Track writes
                if name in ("save_memory", "edit_memory"):
                    write_attempts += 1
                    if "error" not in result:
                        write_count += 1
                elif name == "recall":
                    recall_count += 1

            except Exception as exc:
                result = {"error": f"{type(exc).__name__}: {exc}"}
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result, ensure_ascii=False, default=str),
            })

    return {
        "tool_names": sorted(tool_names),
        "write_count": write_count,
        "write_attempts": write_attempts,
        "recall_count": recall_count,
    }

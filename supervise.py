#!/usr/bin/env python3
"""Offline supervision run — a senior model reviews captured разборы (ADR 0003).

    ./supervise.py                    # review pending cases (needs enough material)
    ./supervise.py --stats            # show what's accumulated, per tenant
    ./supervise.py --tenant <id>      # only this tenant's разборы
    ./supervise.py --model <id>       # critic model (OpenRouter id); default Opus
    ./supervise.py --force            # review even below the material threshold
    ./supervise.py -n 10              # cap how many cases to review this run

The critic judges each case against the *applied skill's own* «критерий
качественного разбора», plus a thin cross-school layer (MITI partnership/empathy,
ICF non-directiveness / follow-through). Findings are itemised and every point must
quote the session — holistic verdicts invite self-preference bias, since coach and
critic are the same model family.
"""

import argparse
import asyncio
import os
import sys

import config
import memory
import supervision

# Cross-family by default: the coach runs on Claude, so a Claude critic partly
# approves its own house style (self-preference bias). A stronger same-family
# critic gives the sharper разбор — override with SUPERVISOR_MODEL when that
# matters more than independence — but the default should not have correlated
# blind spots with the thing it's judging.
SUPERVISOR_MODEL = os.environ.get("SUPERVISOR_MODEL", "google/gemini-2.5-pro")
MIN_CASES = int(os.environ.get("SUPERVISION_MIN_CASES", "20"))

RUBRIC = """Ты — супервизор коуча. Разбираешь работу AI-коуча, чтобы улучшить его
скиллы и обвязку. Ты НЕ оцениваешь клиента и не разбираешь его личность.

Оцени разбор по трём слоям. По КАЖДОМУ пункту дай:
  вердикт (+ / − / н/п), дословную цитату из сессии как основание, и — если «−» —
  что конкретно надо поправить в скилле или обвязке.

Без цитаты пункт не засчитывается. Не давай общей оценки «хорошо/плохо» — только
по пунктам.

СЛОЙ 1 — контракт применённого скилла (главный).
{skill_criterion}

СЛОЙ 2 — партнёрство и эмпатия (кросс-школьно, из MITI):
  2.1 Партнёрство: коуч работает вместе с человеком, а не сверху-вниз?
  2.2 Эмпатия: отражает то, что человек реально принёс, а не свою интерпретацию?

СЛОЙ 3 — выбор оптики (обвязка, не человек):
  3.0 Была ли применена подходящая оптика из каталога? Ниже — что коуч
      подгрузил (или не подгрузил) и что вообще было доступно. Если запрос явно
      попадал в описание неиспользованной оптики — это дефект маршрутизации, а
      не стиля: так и напиши в «ЧТО ПРАВИТЬ». Если базового плейбука хватало —
      скажи прямо, не выдумывай замечание.

КАТАЛОГ ДОСТУПНЫХ ОПТИК:
{catalogue}

СЛОЙ 4 — коучинговые маркеры (ICF):
  4.1 Недирективность: помогает найти ответ, а не выдаёт готовый вывод?
  4.2 Тема клиента: работает с тем, что принёс человек, а не с подменённой темой?
  4.3 Доведение до действия: разбор закончился честным следующим шагом?

В конце — блок «ЧТО ПРАВИТЬ» — 1-3 конкретных пункта: правка скилла, промпта или
обвязки. Если разбор хороший — так и напиши, не выдумывай замечаний."""

FALLBACK_CRITERION = """(Скилл не зафиксирован — оценивай по общему критерию:
разбор даёт ясность, не усиливает внутреннюю войну, опирается на реальность
человека и заканчивается честным следующим шагом.)"""


def _criterion(skill_title):
    """Pull the applied skill's own quality criterion — that's what the coach
    signed up to, so that's what it's judged against."""
    if not skill_title:
        return FALLBACK_CRITERION
    body = memory._load_skill_sync(skill_title)
    if not body:
        return FALLBACK_CRITERION
    marker = "## Критерий качественного разбора"
    if marker not in body:
        return FALLBACK_CRITERION
    section = body.split(marker, 1)[1]
    # stop at the next heading of the same level
    end = section.find("\n## ")
    return marker + (section[:end] if end != -1 else section)


def _catalogue():
    """What the coach could have reached for. Without it the critic can't tell a
    missing optic from an optic that doesn't exist."""
    items = memory._catalog_sync()
    if not items:
        return "  (каталог пуст)"
    return "\n".join(f"  - {i['title']}: {i['description']}" for i in items)


def _case_prompt(case):
    optic = case["skill"] or "ни одной, только базовый плейбук"
    return (
        f"СЕССИЯ (захвачена по признаку: {case['reasons']}; вызванные инструменты: "
        f"{case['tools'] or '—'}; применённая оптика: {optic})\n\n"
        f"--- Что написал человек ---\n{case['user']}\n\n"
        f"--- Что ответил коуч ---\n{case['answer']}\n"
    )


_RATING_LABEL = {"hit": "в точку", "partial": "частично", "miss": "мимо"}


def _headline(case):
    """The case header, with the person's rating next to the tools the coach
    called. That pairing is the whole point: «мимо» when recall was never called
    diagnoses the search, «мимо» with recall in the trace diagnoses the разбор.

    The rating deliberately does not go into _case_prompt. The model judge is
    discredited (R5), which makes the person's rating the one independent signal
    we have — show it to the critic and it anchors on the answer instead of
    judging, and the two stop being comparable."""
    parts = [f"КЕЙС {case['id']}", f"скилл: {case['skill'] or '—'}",
             f"признак: {case['reasons']}", f"инструменты: {case['tools'] or '—'}"]
    if case.get("rating"):
        parts.append(f"человек: {_RATING_LABEL.get(case['rating'], case['rating'])}")
    return " · ".join(parts)


async def _review(client, case, model):
    system = RUBRIC.format(skill_criterion=_criterion(case["skill"]),
                           catalogue=_catalogue())
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": _case_prompt(case)}],
        max_tokens=4000,
    )
    return resp.choices[0].message.content or "(пустой вердикт)"


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="run below the material threshold")
    ap.add_argument("--stats", action="store_true", help="show accumulated material and exit")
    ap.add_argument("--tenant", help="only this tenant's разборы")
    ap.add_argument("--model", default=SUPERVISOR_MODEL, help="critic model (OpenRouter id)")
    ap.add_argument("-n", type=int, default=50, help="max cases this run")
    args = ap.parse_args()

    supervision.init()
    s = supervision.stats()

    if args.stats:
        print(f"Всего кейсов: {s['total']} · ждут разбора: {s['pending']} · "
              f"порог запуска: {MIN_CASES}\n")
        print("По тенантам (всего / ждут разбора):")
        for tenant, tot, waiting in s["by_tenant"]:
            enough = "✓ хватает" if waiting >= MIN_CASES else f"нужно ещё {MIN_CASES - waiting}"
            print(f"  {tenant}  {tot} / {waiting}   → {enough}")
        print("\nПо признаку захвата:")
        for reason, n in s["by_reason"]:
            print(f"  {reason}: {n}")
        return

    # scope the material threshold to the tenant we're actually about to review
    pending_here = args.tenant and next(
        (w for t, _, w in s["by_tenant"] if t == args.tenant), 0)
    pending_count = pending_here if args.tenant else s["pending"]
    if pending_count < MIN_CASES and not args.force:
        scope = f"тенант {args.tenant}" if args.tenant else "всего"
        print(f"Материала мало ({scope}): {pending_count} кейсов (порог {MIN_CASES}). "
              f"Выводы будут шумные — накопи ещё или запусти с --force.")
        return

    cases = supervision.pending(limit=args.n, tenant=args.tenant)
    if not cases:
        print("Нечего разбирать.")
        return

    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY)

    print(f"Супервизия: {len(cases)} кейс(ов), модель {args.model}"
          f"{', тенант ' + args.tenant if args.tenant else ''}\n")
    for case in cases:
        try:
            verdict = await _review(client, case, args.model)
        except Exception as exc:
            print(f"[case {case['id']}] ошибка критика: {exc}", file=sys.stderr)
            continue
        supervision.mark_reviewed(case["id"], verdict)
        print("=" * 70)
        print(_headline(case))
        print("=" * 70)
        print(verdict)
        print()


if __name__ == "__main__":
    asyncio.run(main())

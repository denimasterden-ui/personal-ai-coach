#!/usr/bin/env python3
"""Offline supervision run — a senior model reviews captured разборы (ADR 0003).

    ./supervise.py              # review pending cases (needs enough material)
    ./supervise.py --force      # review even below the material threshold
    ./supervise.py --stats      # just show what's accumulated
    ./supervise.py -n 10        # cap how many cases to review this run

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

SUPERVISOR_MODEL = os.environ.get("SUPERVISOR_MODEL", "anthropic/claude-opus-4.5")
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

СЛОЙ 3 — коучинговые маркеры (ICF):
  3.1 Недирективность: помогает найти ответ, а не выдаёт готовый вывод?
  3.2 Тема клиента: работает с тем, что принёс человек, а не с подменённой темой?
  3.3 Доведение до действия: разбор закончился честным следующим шагом?

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


def _case_prompt(case):
    return (
        f"СЕССИЯ (захвачена по признаку: {case['reasons']}; вызванные инструменты: "
        f"{case['tools'] or '—'})\n\n"
        f"--- Что написал человек ---\n{case['user']}\n\n"
        f"--- Что ответил коуч ---\n{case['answer']}\n"
    )


async def _review(client, case):
    system = RUBRIC.format(skill_criterion=_criterion(case["skill"]))
    resp = await client.chat.completions.create(
        model=SUPERVISOR_MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": _case_prompt(case)}],
        max_tokens=4000,
    )
    return resp.choices[0].message.content or "(пустой вердикт)"


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="run below the material threshold")
    ap.add_argument("--stats", action="store_true", help="show accumulated material and exit")
    ap.add_argument("-n", type=int, default=50, help="max cases this run")
    args = ap.parse_args()

    supervision.init()
    s = supervision.stats()

    if args.stats:
        print(f"Всего кейсов: {s['total']} · ждут разбора: {s['pending']}")
        for reason, n in s["by_reason"]:
            print(f"  {reason}: {n}")
        return

    if s["pending"] < MIN_CASES and not args.force:
        print(f"Материала мало: {s['pending']} кейсов (порог {MIN_CASES}). "
              f"Выводы будут шумные — накопи ещё или запусти с --force.")
        return

    cases = supervision.pending(limit=args.n)
    if not cases:
        print("Нечего разбирать.")
        return

    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY)

    print(f"Супервизия: {len(cases)} кейс(ов), модель {SUPERVISOR_MODEL}\n")
    for case in cases:
        try:
            verdict = await _review(client, case)
        except Exception as exc:
            print(f"[case {case['id']}] ошибка критика: {exc}", file=sys.stderr)
            continue
        supervision.mark_reviewed(case["id"], verdict)
        print("=" * 70)
        print(f"КЕЙС {case['id']} · скилл: {case['skill'] or '—'} · признак: {case['reasons']}")
        print("=" * 70)
        print(verdict)
        print()


if __name__ == "__main__":
    asyncio.run(main())

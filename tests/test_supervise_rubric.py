"""supervise.py — the rubric the critic is actually handed.

The rubric is a format string assembled from several parts; a stray brace or a
missing key turns every review into a KeyError at runtime, long after the cases
were captured. These tests are cheap and catch that at commit time.
"""
import supervise


def test_rubric_renders_with_both_slots():
    out = supervise.RUBRIC.format(skill_criterion="КРИТЕРИЙ", catalogue="  - X: y")
    assert "КРИТЕРИЙ" in out and "  - X: y" in out


def test_rubric_asks_about_optic_selection():
    """Code can't judge which optic fits a request — the critic can, but only if
    the rubric tells it to look. Four разборов routed to no optic at all is how
    this gap surfaced."""
    out = supervise.RUBRIC.format(skill_criterion="—", catalogue="—")
    assert "оптик" in out.lower()


def test_case_prompt_names_the_applied_optic(monkeypatch):
    case = {"reasons": "full", "tools": "recall", "skill": None,
            "user": "вопрос", "answer": "ответ"}
    text = supervise._case_prompt(case)
    assert "ни одной" in text, "critic must see that no optic fired, not a blank"
    case["skill"] = "Identity Alignment v1"
    assert "Identity Alignment v1" in supervise._case_prompt(case)


def test_catalogue_survives_an_empty_skills_dir(monkeypatch):
    monkeypatch.setattr(supervise.memory, "_catalog_sync", lambda: [])
    assert supervise._catalogue()  # must render something, not crash


# ── the person's rating, beside the trace (aicoach-dev#12, п.22) ──────────────


def test_headline_puts_the_rating_next_to_the_tools():
    """«Мимо» при пустом recall и «мимо» при пяти найденных записях — два разных
    диагноза: первое про поиск, второе про разбор. Оба видны только рядом."""
    line = supervise._headline({"id": 7, "skill": None, "reasons": "onboarding",
                                "tools": "recall", "rating": "miss"})
    assert "мимо" in line.lower()
    assert "recall" in line


def test_headline_says_nothing_when_nobody_rated():
    line = supervise._headline({"id": 7, "skill": None, "reasons": "full",
                                "tools": "", "rating": None})
    assert "мимо" not in line.lower() and "точку" not in line.lower()


def test_rating_is_kept_out_of_the_critic_prompt():
    """Deliberate: the model judge was discredited (R5), so the person's rating
    is the one independent signal we have. Handing it to the critic would let it
    anchor on the answer instead of judging, and contaminate the comparison."""
    case = {"reasons": "full", "tools": "recall", "skill": None,
            "user": "вопрос", "answer": "ответ", "rating": "miss"}
    text = supervise._case_prompt(case)
    assert "miss" not in text and "мимо" not in text.lower()

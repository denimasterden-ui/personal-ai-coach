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

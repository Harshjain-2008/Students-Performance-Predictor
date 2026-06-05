"""
src/utils.py
────────────
Shared utility functions used by both app.py (Streamlit)
and predictor.py (terminal). No ML or UI code here.
"""

import os


# ── grade config ──────────────────────────────────────────────
GRADE_CONFIG = {
    "A+": {"min": 90, "color": "#22c55e", "emoji": "🏆", "label": "Excellent"},
    "A":  {"min": 80, "color": "#4ade80", "emoji": "⭐", "label": "Great"},
    "B":  {"min": 70, "color": "#60a5fa", "emoji": "👍", "label": "Good"},
    "C":  {"min": 60, "color": "#facc15", "emoji": "📘", "label": "Average"},
    "D":  {"min": 40, "color": "#fb923c", "emoji": "⚠️", "label": "Below avg"},
    "F":  {"min":  0, "color": "#f87171", "emoji": "❌", "label": "Fail"},
}

FEATURE_META = {
    "study_hours":      {"icon": "📚", "hint": "hours per day (0–12)",   "max": 12.0},
    "previous_score":   {"icon": "📝", "hint": "last exam score (0–100)","max": 100.0},
    "attendance":       {"icon": "🏫", "hint": "percentage (0–100)",     "max": 100.0},
    "sleep_hours":      {"icon": "😴", "hint": "hours per night (0–12)", "max": 12.0},
    "assignments_done": {"icon": "✅", "hint": "number completed",       "max": 20.0},
}


# ── grade helpers ─────────────────────────────────────────────
def get_grade_config(grade: str) -> dict:
    """Return colour/emoji/label config for a grade letter."""
    return GRADE_CONFIG.get(grade, GRADE_CONFIG["F"])


def feature_icon(feature_name: str) -> str:
    """Return an emoji icon for a known feature, or a default."""
    return FEATURE_META.get(feature_name, {}).get("icon", "🔢")


def feature_hint(feature_name: str) -> str:
    """Return a short hint string for a known feature."""
    return FEATURE_META.get(feature_name, {}).get("hint", "numeric value")


# ── file helpers ──────────────────────────────────────────────
def is_number(value: str) -> bool:
    """Return True if string can be parsed as a float."""
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def safe_mkdir(path: str) -> None:
    """Create directory (and parents) if it does not exist."""
    os.makedirs(path, exist_ok=True)


def output_path(input_path: str, suffix: str = "_predictions") -> str:
    """
    Build an output file path from an input path.

    Example
    -------
    output_path("data/students.csv") → "data/students_predictions.csv"
    """
    base, ext = os.path.splitext(input_path)
    if ext.lower() in (".xlsx", ".xls"):
        ext = ".csv"
    return f"{base}{suffix}{ext}"


# ── display helpers ───────────────────────────────────────────
def format_score_summary(score: float, grade: str, pass_fail: str) -> str:
    """
    Return a compact one-line text summary for terminal output.

    Example
    -------
    "Score: 78.5 / 100  |  Grade: B  |  Result: PASS ✓"
    """
    pf_symbol = "✓" if pass_fail == "PASS" else "✗"
    return (
        f"Score: {score} / 100  |  "
        f"Grade: {grade}  |  "
        f"Result: {pass_fail} {pf_symbol}"
    )


def build_sample_dataframe():
    """
    Build and return the built-in 42-row sample dataset as a DataFrame.
    Used when no external file is provided.
    """
    import pandas as pd

    study_hours_list = [
        1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0,
        6.5,7.0,7.5,8.0,8.5,9.0,9.5,10.0,
        1.0,2.8,3.2,4.8,5.8,6.8,7.8,8.8,
        1.2,2.2,3.8,4.2,5.2,6.2,7.2,8.2,9.2,
        1.8,2.6,4.6,5.6,6.6,7.6,3.0
    ]

    rows = []
    for sh in study_hours_list:
        ps  = round(min(100, 35 + sh * 5.5))
        att = round(min(100, 55 + sh * 4.5))
        slp = round(min(10,  5  + sh * 0.55))
        asg = round(min(10,  2  + sh * 0.9))
        fs  = round(min(100, 30 + sh * 7.2 + (ps - 35) * 0.05 + (att - 55) * 0.1), 1)
        rows.append({
            "study_hours":      sh,
            "previous_score":   ps,
            "attendance":       att,
            "sleep_hours":      slp,
            "assignments_done": asg,
            "final_score":      fs,
        })

    return pd.DataFrame(rows)

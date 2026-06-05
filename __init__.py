"""
src/__init__.py
───────────────
Makes `src` a Python package so both app.py and predictor.py
can import from it cleanly:
"""
# from src.model import StudentModel
# from src.utils import build_sample_dataframe


from model import (
    StudentModel,
    load_dataset,
    load_dataset_from_bytes,
    score_to_grade,
    score_to_passfail,
)
from utils import (
    build_sample_dataframe,
    GRADE_CONFIG,
    FEATURE_META,
)

__all__ = [
    "StudentModel",
    "load_dataset",
    "load_dataset_from_bytes",
    "score_to_grade",
    "score_to_passfail",
    "build_sample_dataframe",
    "GRADE_CONFIG",
    "FEATURE_META",
]

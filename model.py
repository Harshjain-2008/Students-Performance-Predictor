"""
src/model.py
────────────
Core ML logic — data loading, training, prediction.
No UI code here. Import this in app.py or predictor.py.
"""

import math
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


# ── constants ─────────────────────────────────────────────────
PREFERRED_TARGETS = [
    "final_score", "score", "marks", "result",
    "final_marks", "exam_score", "total_score",
]
PREFERRED_FEATURES = [
    "study_hours", "previous_score", "attendance",
    "sleep_hours", "assignments_done", "extra_classes",
    "participation", "quiz_score",
]
PASS_THRESHOLD = 40


# ── helpers ───────────────────────────────────────────────────
def score_to_grade(score: float) -> str:
    """Convert numeric score to letter grade."""
    if score >= 90: return "A+"
    if score >= 80: return "A"
    if score >= 70: return "B"
    if score >= 60: return "C"
    if score >= 40: return "D"
    return "F"


def score_to_passfail(score: float) -> str:
    """Return PASS or FAIL string."""
    return "PASS" if score >= PASS_THRESHOLD else "FAIL"


def clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    """Clamp a predicted score to [0, 100]."""
    return max(lo, min(hi, value))


def auto_detect_target(columns: list[str]) -> str | None:
    """Return the most likely target column name from a list."""
    for name in PREFERRED_TARGETS:
        if name in columns:
            return name
    return columns[-1] if columns else None


def auto_detect_features(columns: list[str], target: str) -> list[str]:
    """Return the most likely feature columns (excludes target)."""
    non_target = [c for c in columns if c != target]
    preferred  = [c for c in non_target if c in PREFERRED_FEATURES]
    return preferred if preferred else non_target[:6]


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase + underscore column names."""
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


# ── data loader ───────────────────────────────────────────────
def load_dataset(path: str) -> pd.DataFrame:
    """
    Load a CSV or Excel file and normalise column names.

    Parameters
    ----------
    path : str
        File path to .csv, .xlsx, or .xls

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    ValueError  – unsupported file extension
    FileNotFoundError – file not found
    """
    import os
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file type '{ext}'. Use .csv or .xlsx")

    return normalise_columns(df)


def load_dataset_from_bytes(file_bytes, filename: str) -> pd.DataFrame:
    """
    Load dataset from an in-memory bytes object (for Streamlit uploads).

    Parameters
    ----------
    file_bytes : file-like object
    filename   : original filename (used to detect extension)
    """
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext == "csv":
        df = pd.read_csv(file_bytes)
    elif ext in ("xlsx", "xls"):
        df = pd.read_excel(file_bytes)
    else:
        raise ValueError(f"Unsupported file type '.{ext}'")
    return normalise_columns(df)


# ── model class ───────────────────────────────────────────────
class StudentModel:
    """
    Wraps a scikit-learn Linear Regression pipeline.

    Usage
    -----
    model = StudentModel()
    model.train(df, target="final_score", features=["study_hours", ...])
    score = model.predict({"study_hours": 6, "attendance": 80, ...})
    """

    def __init__(self):
        self._model   = LinearRegression()
        self._scaler  = StandardScaler()
        self.features : list[str] = []
        self.target   : str       = ""
        self.df       : pd.DataFrame | None = None
        self.coef_df  : pd.DataFrame | None = None
        self.is_trained: bool = False

        # performance metrics (set after training)
        self.r2   : float | None = None
        self.mae  : float | None = None
        self.rmse : float | None = None
        self.n_rows: int  = 0

    # ── training ──────────────────────────────────────────────
    def train(
        self,
        df: pd.DataFrame,
        target: str,
        features: list[str],
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> None:
        """
        Train the model on df[features] → df[target].

        Parameters
        ----------
        df           : full dataset
        target       : name of the column to predict
        features     : list of input column names
        test_size    : fraction held out for evaluation (default 0.2)
        random_state : reproducibility seed
        """
        self.df       = df.copy()
        self.target   = target
        self.features = features

        data = df[features + [target]].dropna()
        X, y = data[features], data[target]
        self.n_rows = len(data)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        X_train_s = self._scaler.fit_transform(X_train)
        X_test_s  = self._scaler.transform(X_test)

        self._model.fit(X_train_s, y_train)
        preds = self._model.predict(X_test_s)

        # metrics
        self.mae  = round(mean_absolute_error(y_test, preds), 4)
        self.r2   = round(r2_score(y_test, preds), 4)
        self.rmse = round(math.sqrt(np.mean((y_test - preds) ** 2)), 4)

        # coefficient table
        self.coef_df = pd.DataFrame({
            "Feature":     features,
            "Coefficient": [round(c, 4) for c in self._model.coef_],
            "Direction":   ["↑ Positive" if c > 0 else "↓ Negative"
                            for c in self._model.coef_],
        }).sort_values("Coefficient", key=abs, ascending=False).reset_index(drop=True)

        self.is_trained = True

    # ── prediction ────────────────────────────────────────────
    def predict(self, feature_values: dict) -> float:
        """
        Predict final score for a single student.

        Parameters
        ----------
        feature_values : dict mapping feature name → numeric value
                         Must include all columns in self.features.

        Returns
        -------
        float : predicted score clamped to [0, 100]
        """
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet. Call .train() first.")

        row   = pd.DataFrame([feature_values], columns=self.features)
        row_s = self._scaler.transform(row)
        raw   = self._model.predict(row_s)[0]
        return clamp(round(float(raw), 1))

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict scores for every row in df.
        Adds 'predicted_score', 'predicted_grade', 'pass_fail' columns.

        Missing feature values are filled with column means.
        """
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet. Call .train() first.")

        result = df.copy()
        X      = result[self.features].fillna(result[self.features].mean())
        X_s    = self._scaler.transform(X)
        preds  = self._model.predict(X_s)

        result["predicted_score"] = [clamp(round(float(p), 1)) for p in preds]
        result["predicted_grade"] = result["predicted_score"].apply(score_to_grade)
        result["pass_fail"]       = result["predicted_score"].apply(score_to_passfail)
        return result

    # ── utilities ─────────────────────────────────────────────
    def feature_means(self) -> dict:
        """Return mean value of each feature from the training data."""
        if self.df is None:
            return {}
        return self.df[self.features].mean().to_dict()

    def metrics_dict(self) -> dict:
        """Return a dict of all evaluation metrics."""
        return {
            "r2":       self.r2,
            "mae":      self.mae,
            "rmse":     self.rmse,
            "n_rows":   self.n_rows,
            "features": len(self.features),
            "target":   self.target,
        }

    def summary(self) -> str:
        """Return a human-readable one-line summary."""
        if not self.is_trained:
            return "Model not trained."
        return (
            f"LinearRegression | target='{self.target}' | "
            f"features={self.features} | "
            f"R²={self.r2} | MAE={self.mae} marks | rows={self.n_rows}"
        )

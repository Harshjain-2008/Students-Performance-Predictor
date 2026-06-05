"""
predictor.py
────────────
Terminal / CLI entry point.
All ML logic lives in src/model.py — this file handles terminal I/O only.

Run:
    python predictor.py
"""

import os
import sys

from model import StudentModel, load_dataset, auto_detect_target, auto_detect_features
from utils import (
    build_sample_dataframe,
    feature_hint,
    format_score_summary,
    output_path,
    is_number,
)

# ── terminal helpers ──────────────────────────────────────────
WIDTH = 64

def banner():
    os.system("cls" if os.name == "nt" else "clear")
    print("=" * WIDTH)
    print("   STUDENT PERFORMANCE PREDICTOR  v2.0".center(WIDTH))
    print("   src/model.py  ·  src/utils.py".center(WIDTH))
    print("=" * WIDTH)

def hr(char="─"):
    print(char * WIDTH)

def section(title: str):
    hr(); print(f"  {title}"); hr()

def prompt(msg: str, default=None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    return input(f"  {msg}{suffix}: ").strip()

def get_number(label: str, last=None) -> float:
    hint = feature_hint(label)
    while True:
        raw = prompt(f"{label}  ({hint})", last)
        if raw == "" and last is not None:
            return float(last)
        if is_number(raw):
            return float(raw)
        print("  [!] Please enter a valid number.")


# ── column picker ─────────────────────────────────────────────
def pick_columns(df):
    cols = list(df.columns)
    section("COLUMN SELECTION")
    for i, c in enumerate(cols):
        print(f"    {i+1:2}. {c}")
    hr()

    auto_tgt = auto_detect_target(cols)
    print(f"\n  Which column is the TARGET (score to predict)?")
    if auto_tgt:
        print(f"  Auto-detected: '{auto_tgt}'  (press Enter to accept)")
    inp = prompt("Column name or number", auto_tgt)
    target = cols[int(inp) - 1] if is_number(inp) else (inp if inp in cols else auto_tgt)

    auto_feats = auto_detect_features(cols, target)
    print(f"\n  Which columns are FEATURES?")
    print(f"  Auto-selected: {auto_feats}")
    print(f"  Press Enter to accept, or type comma-separated names/numbers.")
    inp = prompt("Features", "")
    if inp:
        parts    = [p.strip() for p in inp.split(",")]
        features = [cols[int(p)-1] if is_number(p) else p for p in parts if p in cols or is_number(p)]
    else:
        features = auto_feats

    return target, features


# ── modes ─────────────────────────────────────────────────────
def mode_quick_predict(m: StudentModel):
    section("QUICK PREDICT — by Study Hours")
    print("  Other features use dataset averages.\n")
    means = m.feature_means()
    last  = {}
    while True:
        hrs = get_number("study_hours", last.get("study_hours", 5.0))
        last["study_hours"] = hrs
        row   = {**means, "study_hours": hrs}
        row   = {f: row.get(f, 0) for f in m.features}
        score = m.predict(row)

        hr("═")
        print(f"  Study hours   : {hrs} hrs/day")
        print(f"  {format_score_summary(score, *_grade_pf(score))}")
        hr("═")

        if prompt("Try another value? (y/n)", "y").lower() != "y":
            break


def mode_full_predict(m: StudentModel):
    section("FULL PREDICTION — All Features")
    last = {}
    while True:
        row = {}
        for feat in m.features:
            row[feat] = get_number(feat, last.get(feat))
        last = row.copy()
        score = m.predict(row)

        hr("═")
        print(f"  {format_score_summary(score, *_grade_pf(score))}")
        hr("═")

        if prompt("Predict for another student? (y/n)", "y").lower() != "y":
            break


def mode_batch_predict(m: StudentModel):
    section("BATCH PREDICTION — from File")
    path = prompt("Enter path to CSV/Excel file")
    if not os.path.exists(path):
        print("  [!] File not found.")
        return
    try:
        df     = load_dataset(path)
        miss   = [f for f in m.features if f not in df.columns]
        if miss:
            print(f"  [!] Missing columns: {miss}")
            return

        result = m.predict_dataframe(df)
        out    = output_path(path)
        result.to_csv(out, index=False)

        print(f"\n  ✓ Results saved to: {out}")
        print(f"  Pass rate : {(result['pass_fail']=='PASS').mean()*100:.1f}%")
        print(f"  Avg score : {result['predicted_score'].mean():.1f}")
        from tabulate import tabulate
        print(tabulate(
            result[["predicted_score", "predicted_grade", "pass_fail"]].head(10),
            headers="keys", tablefmt="simple", showindex=True,
        ))
    except Exception as e:
        print(f"  [!] Error: {e}")


def mode_load_data(m: StudentModel):
    section("LOAD NEW DATASET & RETRAIN")
    path = prompt("Enter path to CSV or Excel file")
    if not os.path.exists(path):
        print("  [!] File not found.")
        return False
    try:
        df     = load_dataset(path)
        print(f"\n  Loaded {len(df)} rows × {len(df.columns)} columns")
        target, features = pick_columns(df)
        section("TRAINING MODEL")
        m.train(df, target, features)
        print("  ✓ Model trained!")
        _show_stats(m)
        return True
    except Exception as e:
        print(f"  [!] Error: {e}")
        return False


def _show_stats(m: StudentModel):
    from tabulate import tabulate
    section("MODEL PERFORMANCE")
    rows = [
        ["R²",       f"{m.r2:.4f}  ({m.r2*100:.1f}% variance explained)"],
        ["MAE",      f"{m.mae:.2f} marks"],
        ["RMSE",     f"{m.rmse:.2f} marks"],
        ["Rows",     m.n_rows],
        ["Features", len(m.features)],
    ]
    print(tabulate(rows, tablefmt="simple"))
    hr()
    print("  Feature coefficients:\n")
    print(tabulate(m.coef_df.values.tolist(),
                   headers=["Feature","Coefficient","Direction"],
                   tablefmt="simple", floatfmt=".4f"))


def _grade_pf(score):
    from model import score_to_grade, score_to_passfail
    return score_to_grade(score), score_to_passfail(score)


# ── main ──────────────────────────────────────────────────────
def main():
    banner()
    m = StudentModel()

    section("WELCOME")
    print("  1. Load YOUR CSV / Excel file")
    print("  2. Use built-in sample data\n")
    choice = prompt("Choose", "2")

    if choice == "1":
        ok = mode_load_data(m)
        if not ok:
            choice = "2"

    if choice != "1":
        df     = build_sample_dataframe()
        target = "final_score"
        feats  = ["study_hours", "previous_score", "attendance",
                  "sleep_hours", "assignments_done"]
        m.train(df, target, feats)
        print("\n  ✓ Sample data loaded & model trained!")
        _show_stats(m)

    while True:
        section("MAIN MENU")
        print("  1.  Quick Predict  — study hours only")
        print("  2.  Full Predict   — all features")
        print("  3.  Batch Predict  — from file")
        print("  4.  Load New Data  — retrain on new dataset")
        print("  5.  Model Stats    — accuracy & coefficients")
        print("  6.  Exit\n")

        opt = prompt("Choose", "1")
        if   opt == "1": mode_quick_predict(m)
        elif opt == "2": mode_full_predict(m)
        elif opt == "3": mode_batch_predict(m)
        elif opt == "4": mode_load_data(m)
        elif opt == "5": _show_stats(m)
        elif opt == "6": print("\n  Goodbye! 📚\n"); break
        else:            print("  [!] Invalid option.")

        input("\n  Press Enter to continue...")
        banner()


if __name__ == "__main__":
    main()

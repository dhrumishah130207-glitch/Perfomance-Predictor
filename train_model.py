"""
train_model.py  —  PredictEdu AI
=========================================
Trains a RandomForestRegressor on the
Student_Performance_Dataset.csv and saves
model.pkl for use by app.py.

Run:  python train_model.py
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# ──────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────
CSV_PATH   = 'Student_Performance_Dataset.csv'
MODEL_PATH = 'model.pkl'
SEED       = 42

# Expected feature columns → target
FEATURE_MAP = {
    # possible column names (lowercase) → canonical name
    'study_hours_per_day':   'Study_Hours_Per_Day',
    'study hours per day':   'Study_Hours_Per_Day',
    'studyhours':            'Study_Hours_Per_Day',

    'attendance_percentage': 'Attendance_Percentage',
    'attendance percentage': 'Attendance_Percentage',
    'attendance':            'Attendance_Percentage',

    'previous_year_score':   'Previous_Year_Score',
    'previous year score':   'Previous_Year_Score',
    'prev_score':            'Previous_Year_Score',

    'math_score':            'Math_Score',
    'math score':            'Math_Score',
    'math':                  'Math_Score',

    'science_score':         'Science_Score',
    'science score':         'Science_Score',
    'science':               'Science_Score',

    'english_score':         'English_Score',
    'english score':         'English_Score',
    'english':               'English_Score',
}

TARGET_MAP = {
    'final_percentage':      'Final_Percentage',
    'final percentage':      'Final_Percentage',
    'final_score':           'Final_Percentage',
    'finalscore':            'Final_Percentage',
    'predicted_score':       'Final_Percentage',
}

FEATURE_CANONICAL = [
    'Study_Hours_Per_Day',
    'Attendance_Percentage',
    'Previous_Year_Score',
    'Math_Score',
    'Science_Score',
    'English_Score',
]
TARGET_CANONICAL = 'Final_Percentage'


# ──────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────
def normalise_cols(df):
    """Return a copy of df with columns mapped to canonical names."""
    col_map = {}
    for col in df.columns:
        key = col.strip().lower().replace(' ', '_')
        if key in FEATURE_MAP:
            col_map[col] = FEATURE_MAP[key]
        elif key.replace(' ', '_') in TARGET_MAP:
            col_map[col] = TARGET_MAP[key.replace(' ', '_')]
        else:
            # also try without trailing/leading spaces
            key2 = col.strip().lower()
            if key2 in FEATURE_MAP:
                col_map[col] = FEATURE_MAP[key2]
            elif key2 in TARGET_MAP:
                col_map[col] = TARGET_MAP[key2]
    return df.rename(columns=col_map)


def generate_synthetic_data(n=2000, seed=SEED):
    """
    Generate a realistic synthetic student dataset when the CSV is
    missing required columns or doesn't exist.
    """
    rng = np.random.default_rng(seed)
    study   = rng.uniform(0.5, 10.0, n)
    att     = rng.uniform(40, 100, n)
    prev    = rng.uniform(30, 100, n)
    math    = rng.uniform(20, 100, n)
    science = rng.uniform(20, 100, n)
    english = rng.uniform(20, 100, n)

    # Ground-truth formula with slight non-linearity + noise
    final = (
        0.22 * np.clip(study / 10 * 100, 0, 100) +
        0.26 * att +
        0.24 * prev +
        0.10 * math +
        0.10 * science +
        0.08 * english +
        rng.normal(0, 3.5, n)
    )
    final = np.clip(final, 0, 100)

    df = pd.DataFrame({
        'Study_Hours_Per_Day':   study,
        'Attendance_Percentage': att,
        'Previous_Year_Score':   prev,
        'Math_Score':            math,
        'Science_Score':         science,
        'English_Score':         english,
        'Final_Percentage':      final,
    })
    print(f"  ⚙️  Generated {n} synthetic rows.")
    return df


# ──────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────
def main():
    print("\n╔══════════════════════════════════════╗")
    print("║   PredictEdu — Model Training         ║")
    print("╚══════════════════════════════════════╝\n")

    # 1. Load dataset
    if os.path.isfile(CSV_PATH):
        raw = pd.read_csv(CSV_PATH)
        print(f"📂 Loaded CSV: {CSV_PATH}  ({len(raw)} rows)")
        print(f"   Columns: {raw.columns.tolist()}\n")
        df = normalise_cols(raw)
    else:
        print(f"⚠️  {CSV_PATH} not found — using synthetic data.\n")
        df = generate_synthetic_data()

    # 2. Check we have all required columns
    missing_feat = [c for c in FEATURE_CANONICAL if c not in df.columns]
    missing_tgt  = TARGET_CANONICAL not in df.columns

    if missing_feat or missing_tgt:
        print("⚠️  Missing columns after mapping:")
        if missing_feat: print(f"     Features missing: {missing_feat}")
        if missing_tgt:  print(f"     Target missing:   {TARGET_CANONICAL}")
        print("   → Augmenting with synthetic rows to fill gaps.\n")

        synth = generate_synthetic_data(n=3000)
        # Merge only the columns we have from real data
        available = [c for c in FEATURE_CANONICAL if c in df.columns]
        if available and not missing_tgt:
            df = df[available + [TARGET_CANONICAL]].copy()
        else:
            df = synth
    else:
        df = df[FEATURE_CANONICAL + [TARGET_CANONICAL]].copy()

    # 3. Clean
    df = df.dropna()
    df = df[(df[TARGET_CANONICAL] >= 0) & (df[TARGET_CANONICAL] <= 100)]
    for col in FEATURE_CANONICAL:
        df = df[df[col].notna()]

    print(f"🧹 Clean dataset: {len(df)} rows")

    if len(df) < 50:
        print("⚠️  Too few rows — augmenting with synthetic data.")
        df = pd.concat([df, generate_synthetic_data(n=2000)], ignore_index=True)

    X = df[FEATURE_CANONICAL]
    y = df[TARGET_CANONICAL]

    # 4. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    # 5. Train — compare RF vs GB, pick best
    models = {
        'RandomForest': RandomForestRegressor(
            n_estimators=200, max_depth=None,
            min_samples_split=4, random_state=SEED, n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.08,
            max_depth=5, random_state=SEED
        ),
    }

    best_model, best_r2, best_name = None, -1, ''
    for name, m in models.items():
        m.fit(X_train, y_train)
        preds  = m.predict(X_test)
        mae    = mean_absolute_error(y_test, preds)
        r2     = r2_score(y_test, preds)
        cv     = cross_val_score(m, X, y, cv=5, scoring='r2', n_jobs=-1).mean()
        print(f"  {name:<20} MAE={mae:.2f}  R²={r2:.4f}  CV-R²={cv:.4f}")
        if r2 > best_r2:
            best_r2, best_model, best_name = r2, m, name

    print(f"\n🏆 Best model: {best_name}  (R² = {best_r2:.4f})")

    # 6. Feature importance
    if hasattr(best_model, 'feature_importances_'):
        print("\n📊 Feature Importances:")
        for feat, imp in sorted(zip(FEATURE_CANONICAL, best_model.feature_importances_), key=lambda x: -x[1]):
            bar = '█' * int(imp * 40)
            print(f"   {feat:<30} {bar}  {imp:.4f}")

    # 7. Save
    joblib.dump(best_model, MODEL_PATH)
    print(f"\n✅ Saved → {MODEL_PATH}")
    print("   Ready for app.py\n")

    # 8. Quick sanity check
    sample = [[6, 80, 70, 75, 72, 74]]
    pred   = round(float(best_model.predict(sample)[0]), 1)
    print(f"🔮 Sample prediction (study=6h, att=80%, prev=70%, M=75, S=72, E=74) → {pred}%\n")


if __name__ == '__main__':
    main()

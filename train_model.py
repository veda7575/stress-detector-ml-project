"""
Student Stress Detector - ML Training Pipeline
Trains on 3 Kaggle datasets:
  1. StressLevelDataset.csv   → Primary model (1100 rows, 21 features)
  2. Stress_Dataset.csv       → Secondary model (843 rows, 26 features)
  3. Student_Mental_health.csv → Feature engineering for mental health score
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

STRESS_LABELS = {0: "Low", 1: "Moderate", 2: "High"}

# ────────────────────────────────────────────────
# 1. Load & preprocess each dataset
# ────────────────────────────────────────────────

def load_stress_level_dataset():
    """Primary dataset: StressLevelDataset.csv — cleanest, 21 features"""
    df = pd.read_csv(os.path.join(DATA_DIR, "StressLevelDataset.csv"))
    print(f"[Dataset 3] StressLevelDataset: {df.shape[0]} rows, {df.shape[1]} cols")
    print(f"  Stress distribution:\n{df['stress_level'].value_counts().sort_index().to_string()}\n")
    X = df.drop("stress_level", axis=1)
    y = df["stress_level"]
    return X, y, df.columns.tolist()[:-1]


def load_stress_survey_dataset():
    """Secondary dataset: Stress_Dataset.csv — 26 features, survey style"""
    df = pd.read_csv(os.path.join(DATA_DIR, "Stress_Dataset.csv"))
    stress_map = {
        "No Stress - Currently experiencing minimal to no stress.": 0,
        "Eustress (Positive Stress) - Stress that motivates and enhances performance.": 1,
        "Distress (Negative Stress) - Stress that causes anxiety and impairs well-being.": 2,
    }
    df["stress_label"] = df["Which type of stress do you primarily experience?"].map(stress_map)
    df = df.dropna(subset=["stress_label"])
    df["stress_label"] = df["stress_label"].astype(int)
    feature_cols = [c for c in df.columns if c not in
                    ["Which type of stress do you primarily experience?", "stress_label"]]
    print(f"[Dataset 2] Stress_Dataset: {df.shape[0]} rows, {len(feature_cols)} features")
    print(f"  Stress distribution:\n{df['stress_label'].value_counts().sort_index().to_string()}\n")
    return df[feature_cols], df["stress_label"], feature_cols


def load_mental_health_dataset():
    """Dataset 1: Student_Mental_health.csv — encodes mental health indicators"""
    df = pd.read_csv(os.path.join(DATA_DIR, "Student_Mental_health.csv"))
    yn = {"Yes": 1, "No": 0}
    df["depression_flag"] = df["Do you have Depression?"].map(yn).fillna(0)
    df["anxiety_flag"]    = df["Do you have Anxiety?"].map(yn).fillna(0)
    df["panic_flag"]      = df["Do you have Panic attack?"].map(yn).fillna(0)
    df["sought_help"]     = df["Did you seek any specialist for a treatment?"].map(yn).fillna(0)
    cgpa_map = {"3.50 - 4.00": 4, "3.00 - 3.49": 3, "2.50 - 2.99": 2,
                "2.00 - 2.49": 1, "0 - 1.99": 0}
    df["cgpa_score"] = df["What is your CGPA?"].map(
        lambda x: next((v for k, v in cgpa_map.items() if k in str(x)), 2))
    year_map = {"year 1": 1, "Year 1": 1, "year 2": 2, "Year 2": 2,
                "year 3": 3, "Year 3": 3, "year 4": 4, "Year 4": 4}
    df["study_year"] = df["Your current year of Study"].map(year_map).fillna(2)
    df["mental_score"] = df["depression_flag"] + df["anxiety_flag"] + df["panic_flag"]
    df["stress_proxy"] = (df["mental_score"] >= 2).astype(int) * 2 + \
                         (df["mental_score"] == 1).astype(int)
    features = ["depression_flag","anxiety_flag","panic_flag","sought_help",
                 "cgpa_score","study_year","mental_score"]
    print(f"[Dataset 1] Student_Mental_health: {df.shape[0]} rows, mental health features extracted")
    print(f"  Mental score distribution:\n{df['mental_score'].value_counts().sort_index().to_string()}\n")
    return df[features], df["stress_proxy"]


# ────────────────────────────────────────────────
# 2. Train models
# ────────────────────────────────────────────────

def train_primary_model(X, y):
    """Train Random Forest + Gradient Boosting ensemble on StressLevelDataset"""
    print("=" * 55)
    print("TRAINING PRIMARY MODEL (StressLevelDataset)")
    print("=" * 55)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=14, min_samples_split=4,
        random_state=42, class_weight="balanced", n_jobs=-1)

    gb = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.08,
        subsample=0.85, random_state=42)

    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("gb", gb)],
        voting="soft", weights=[2, 1])

    ensemble.fit(X_train, y_train)
    y_pred = ensemble.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc:.4f} ({acc*100:.1f}%)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=["Low", "Moderate", "High"]))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(ensemble, X, y, cv=cv, scoring="accuracy")
    print(f"5-Fold CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Feature importance from RF component
    rf.fit(X_train, y_train)
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=False)
    print("\nTop 10 Feature Importances:")
    print(importances.head(10).to_string())

    return ensemble, importances, {
        "accuracy": round(float(acc), 4),
        "cv_mean": round(float(cv_scores.mean()), 4),
        "cv_std": round(float(cv_scores.std()), 4),
        "features": X.columns.tolist(),
        "feature_importances": importances.to_dict(),
    }


def train_secondary_model(X, y):
    """Train on survey-style Stress_Dataset"""
    print("\n" + "=" * 55)
    print("TRAINING SECONDARY MODEL (Stress_Dataset)")
    print("=" * 55)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=42,
        class_weight="balanced", n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc:.4f} ({acc*100:.1f}%)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=["Low/None", "Eustress", "Distress"]))

    cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    print(f"5-Fold CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return model, {
        "accuracy": round(float(acc), 4),
        "cv_mean": round(float(cv_scores.mean()), 4),
        "features": X.columns.tolist(),
    }


# ────────────────────────────────────────────────
# 3. Inference helper
# ────────────────────────────────────────────────

class StressPredictor:
    """
    High-level wrapper for inference.
    Maps the app's 16 user inputs → StressLevelDataset feature space → prediction.
    """
    FEATURE_NAMES = [
        "anxiety_level", "self_esteem", "mental_health_history", "depression",
        "headache", "blood_pressure", "sleep_quality", "breathing_problem",
        "noise_level", "living_conditions", "safety", "basic_needs",
        "academic_performance", "study_load", "teacher_student_relationship",
        "future_career_concerns", "social_support", "peer_pressure",
        "extracurricular_activities", "bullying"
    ]

    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    def from_survey(self, inputs: dict) -> dict:
        """
        inputs keys (all 0-5 or as noted):
          sleep_hours (2-10), cgpa (0-10), study_load (1-5), attendance (1-4),
          screen_hours (0-12), social_isolation (1-4), exercise (1-4),
          weight_change (0-3), anxiety (1-5), depression_flag (0-4),
          concentration (0-4), panic (0-4), peer_pressure (1-5),
          home_stress (1-4), relationship_stress (0-4), financial (0-4)
        """
        def scale(v, lo, hi, out_lo=0, out_hi=5):
            return round(out_lo + (v - lo) / max(hi - lo, 1) * (out_hi - out_lo), 2)

        s = inputs
        feat = {
            "anxiety_level":              scale(s.get("anxiety", 2), 1, 5, 0, 21),
            "self_esteem":                scale(s.get("cgpa", 7.5), 0, 10, 5, 30),
            "mental_health_history":      1 if s.get("depression_flag", 0) >= 2 else 0,
            "depression":                 scale(s.get("depression_flag", 0), 0, 4, 0, 15),
            "headache":                   scale(s.get("anxiety", 2), 1, 5, 0, 5),
            "blood_pressure":             1 if s.get("anxiety", 2) >= 4 else 0,
            "sleep_quality":              scale(s.get("sleep_hours", 7), 2, 10, 0, 5),
            "breathing_problem":          scale(s.get("panic", 0), 0, 4, 0, 5),
            "noise_level":                scale(s.get("home_stress", 2), 1, 4, 0, 5),
            "living_conditions":          scale(5 - s.get("home_stress", 2), 1, 4, 0, 5),
            "safety":                     3,
            "basic_needs":                scale(4 - s.get("financial", 0), 0, 4, 0, 5),
            "academic_performance":       scale(s.get("cgpa", 7.5), 0, 10, 0, 5),
            "study_load":                 scale(s.get("study_load", 3), 1, 5, 0, 5),
            "teacher_student_relationship": 3,
            "future_career_concerns":     scale(s.get("study_load", 3), 1, 5, 0, 5),
            "social_support":             scale(5 - s.get("social_isolation", 2), 1, 4, 0, 5),
            "peer_pressure":              scale(s.get("peer_pressure", 2), 1, 5, 0, 5),
            "extracurricular_activities": scale(5 - s.get("exercise", 2), 1, 4, 0, 5),
            "bullying":                   scale(s.get("peer_pressure", 2), 1, 5, 0, 5),
        }

        X = pd.DataFrame([feat])[self.FEATURE_NAMES]
        pred = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0]

        stress_pct = round(int(pred) / 2 * 100 + (proba[2] * 30), 1)
        stress_pct = min(100, max(0, stress_pct))

        return {
            "stress_level": int(pred),
            "stress_label": STRESS_LABELS[int(pred)],
            "stress_pct": stress_pct,
            "confidence": round(float(proba.max()) * 100, 1),
            "probabilities": {
                "Low": round(float(proba[0]) * 100, 1),
                "Moderate": round(float(proba[1]) * 100, 1),
                "High": round(float(proba[2]) * 100, 1),
            },
            "feature_values": feat,
        }


# ────────────────────────────────────────────────
# 4. Main — train, save, verify
# ────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  STUDENT STRESS DETECTOR — ML TRAINING PIPELINE")
    print("=" * 55 + "\n")

    # Load datasets
    X3, y3, feat3 = load_stress_level_dataset()
    X2, y2, feat2 = load_stress_survey_dataset()
    X1, y1       = load_mental_health_dataset()

    # Train primary model
    primary_model, importances, primary_meta = train_primary_model(X3, y3)

    # Train secondary model
    secondary_model, secondary_meta = train_secondary_model(X2, y2)

    # Save models
    joblib.dump(primary_model,   os.path.join(MODEL_DIR, "primary_model.pkl"))
    joblib.dump(secondary_model, os.path.join(MODEL_DIR, "secondary_model.pkl"))
    joblib.dump(importances,     os.path.join(MODEL_DIR, "feature_importances.pkl"))

    # Save metadata
    meta = {
        "primary": primary_meta,
        "secondary": secondary_meta,
        "stress_labels": STRESS_LABELS,
        "datasets_used": [
            "StressLevelDataset.csv (1100 rows, primary)",
            "Stress_Dataset.csv (843 rows, secondary)",
            "Student_Mental_health.csv (101 rows, feature engineering)"
        ]
    }
    with open(os.path.join(MODEL_DIR, "model_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\n" + "=" * 55)
    print("MODELS SAVED")
    print("=" * 55)
    print(f"  primary_model.pkl      → {primary_meta['accuracy']*100:.1f}% test accuracy")
    print(f"  secondary_model.pkl    → {secondary_meta['accuracy']*100:.1f}% test accuracy")
    print(f"  feature_importances.pkl")
    print(f"  model_metadata.json")

    # Quick inference test
    print("\n" + "=" * 55)
    print("INFERENCE TEST")
    print("=" * 55)
    predictor = StressPredictor(os.path.join(MODEL_DIR, "primary_model.pkl"))
    test_cases = [
        {"sleep_hours": 8, "cgpa": 8.8, "study_load": 2, "anxiety": 1,
         "depression_flag": 0, "social_isolation": 1, "exercise": 1,
         "peer_pressure": 1, "home_stress": 1, "financial": 0,
         "panic": 0, "concentration": 0, "weight_change": 0,
         "relationship_stress": 0, "screen_hours": 2, "attendance": 1},
        {"sleep_hours": 5, "cgpa": 6.2, "study_load": 4, "anxiety": 4,
         "depression_flag": 3, "social_isolation": 3, "exercise": 3,
         "peer_pressure": 4, "home_stress": 3, "financial": 3,
         "panic": 3, "concentration": 3, "weight_change": 2,
         "relationship_stress": 3, "screen_hours": 8, "attendance": 3},
    ]
    labels = ["Healthy student (expected: Low)", "Stressed student (expected: High)"]
    for label, tc in zip(labels, test_cases):
        result = predictor.from_survey(tc)
        print(f"\n  {label}")
        print(f"    Prediction: {result['stress_label']} ({result['stress_pct']}% stress)")
        print(f"    Confidence: {result['confidence']}%")
        print(f"    Proba: {result['probabilities']}")

    print("\n✓ Training complete!\n")

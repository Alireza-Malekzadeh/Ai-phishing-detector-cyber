"""
Email Phishing Detector - Model Training
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src import extract_features, EmailFeatures

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


def load_dataset(csv_path: str):
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} emails | Phishing: {df['label'].sum()} | Legit: {(df['label']==0).sum()}")

    print("Extracting email features...")
    rows = []
    for raw in df["raw_email"]:
        try:
            f = extract_features(str(raw))
            rows.append(f.to_list())
        except Exception:
            rows.append([0] * len(EmailFeatures.feature_names()))

    X = pd.DataFrame(rows, columns=EmailFeatures.feature_names())
    y = df["label"]
    return X, y


def build_pipelines():
    return {
        "naive_bayes": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GaussianNB())
        ]),
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "random_forest": Pipeline([
            ("clf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
        ]),
        "gradient_boosting": Pipeline([
            ("clf", GradientBoostingClassifier(n_estimators=200, random_state=42))
        ]),
    }


def train(csv_path: str, experiment_name: str = "email-phishing-detector"):
    mlflow.set_experiment(experiment_name)

    X, y = load_dataset(csv_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipelines = build_pipelines()
    results = {}
    best_model, best_auc, best_name = None, 0.0, ""

    for name, pipeline in pipelines.items():
        with mlflow.start_run(run_name=name):
            print(f"\nTraining {name}...")
            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
            report = classification_report(y_test, y_pred, output_dict=True)

            print(classification_report(y_test, y_pred, target_names=["Legitimate", "Phishing"]))
            print(f"ROC-AUC: {auc:.4f}")

            mlflow.log_metric("auc", auc)
            mlflow.sklearn.log_model(pipeline, name)

            results[name] = {"auc": auc, "y_pred": y_pred, "y_prob": y_prob}

            if auc > best_auc:
                best_auc = auc
                best_model = pipeline
                best_name = name

    best_path = os.path.join(MODELS_DIR, "email_detector_best.joblib")
    joblib.dump(best_model, best_path)
    print(f"\n✅ Best: {best_name} (AUC={best_auc:.4f}) → {best_path}")

    # ROC plot
    plt.figure(figsize=(8, 6))
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        plt.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("ROC Curves — Email Phishing Detection")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR, "roc_curves_email.png"), dpi=150)
    plt.close()

    return best_model, results


if __name__ == "__main__":
    import sys
    csv = sys.argv[1] if len(sys.argv) > 1 else "data/processed/email_dataset.csv"
    train(csv)

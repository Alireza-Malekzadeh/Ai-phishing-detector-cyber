"""
URL Phishing Detector - Model Training
Trains and evaluates ML models on the prebuilt URL feature dataset.

Dataset: GregaVrbancic/Phishing-Dataset (58,645 URLs, 111 features)
Run with: python3 -m src.url_detector.trainer
"""

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODELS_DIR = "models"
DATASET_PATH = "data/processed/url_dataset_prebuilt.csv"
os.makedirs(MODELS_DIR, exist_ok=True)


def load_dataset(csv_path: str):
    """Load the preprocessed URL dataset."""
    df = pd.read_csv(csv_path).dropna()
    print(f"Loaded {len(df)} records | "
          f"Phishing: {df['label'].sum()} | "
          f"Legit: {(df['label'] == 0).sum()}")
    X = df.drop("label", axis=1)
    y = df["label"]
    return X, y


def build_pipelines() -> dict:
    """Define candidate model pipelines."""
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "random_forest": Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=200, random_state=42, n_jobs=-1))
        ]),
        "gradient_boosting": Pipeline([
            ("clf", GradientBoostingClassifier(
                n_estimators=200, random_state=42))
        ]),
    }


def plot_confusion_matrix(y_test, y_pred, model_name: str, save_path: str):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Legitimate", "Phishing"],
                yticklabels=["Legitimate", "Phishing"])
    plt.title(f"Confusion Matrix — {model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved confusion matrix → {save_path}")


def plot_roc_curves(results: dict, y_test, save_path: str):
    plt.figure(figsize=(8, 6))
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        plt.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — URL Phishing Detection")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved ROC curves → {save_path}")


def train(csv_path: str = DATASET_PATH,
          experiment_name: str = "url-phishing-detector"):
    """
    Full training pipeline with MLflow tracking.
    Trains multiple models, compares them, saves the best one.
    """
    print("=" * 55)
    print("  URL Phishing Detector — Training Pipeline")
    print("=" * 55)

    mlflow.set_experiment(experiment_name)

    X, y = load_dataset(csv_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}\n")

    pipelines = build_pipelines()
    results = {}
    best_model = None
    best_auc = 0.0
    best_name = ""

    for name, pipeline in pipelines.items():
        print(f"Training {name}...")
        with mlflow.start_run(run_name=name):
            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
            report = classification_report(
                y_test, y_pred,
                target_names=["Legitimate", "Phishing"],
                output_dict=True
            )

            print(classification_report(
                y_test, y_pred,
                target_names=["Legitimate", "Phishing"]
            ))
            print(f"  ROC-AUC: {auc:.4f}\n")

            mlflow.log_metric("auc", auc)
            mlflow.log_metric("f1_phishing", report["Phishing"]["f1-score"])
            mlflow.log_metric("precision_phishing", report["Phishing"]["precision"])
            mlflow.log_metric("recall_phishing", report["Phishing"]["recall"])
            mlflow.sklearn.log_model(pipeline, name)

            cm_path = os.path.join(MODELS_DIR, f"cm_{name}.png")
            plot_confusion_matrix(y_test, y_pred, name, cm_path)
            mlflow.log_artifact(cm_path)

            results[name] = {"auc": auc, "y_pred": y_pred, "y_prob": y_prob}

            if auc > best_auc:
                best_auc = auc
                best_model = pipeline
                best_name = name

    best_path = os.path.join(MODELS_DIR, "url_detector_best.joblib")
    joblib.dump(best_model, best_path)
    print(f"✅ Best model: {best_name} (AUC={best_auc:.4f})")
    print(f"   Saved → {best_path}")

    roc_path = os.path.join(MODELS_DIR, "roc_curves_url.png")
    plot_roc_curves(results, y_test, roc_path)

    return best_model, results


if __name__ == "__main__":
    import sys
    csv = sys.argv[1] if len(sys.argv) > 1 else DATASET_PATH
    train(csv)
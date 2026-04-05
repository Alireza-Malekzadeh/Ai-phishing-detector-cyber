"""
URL Phishing Detector - Model Training
Trains and evaluates ML models on extracted URL features.
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
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.url_detector.feature_extractor import extract_features, URLFeatures

MODELS_DIR = "../../models"
os.makedirs(MODELS_DIR, exist_ok=True)


def load_dataset(csv_path: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load a CSV dataset with columns: 'url' and 'label' (1=phishing, 0=legitimate).
    Returns features matrix X and labels y.
    """
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records | Phishing: {df['label'].sum()} | Legit: {(df['label'] == 0).sum()}")

    print("Extracting URL features...")
    feature_rows = []
    for url in df["url"]:
        feats = extract_features(str(url))
        feature_rows.append(feats.to_list())

    X = pd.DataFrame(feature_rows, columns=URLFeatures.feature_names())
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
            ("clf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
        ]),
        "gradient_boosting": Pipeline([
            ("clf", GradientBoostingClassifier(n_estimators=200, random_state=42))
        ]),
    }


def evaluate_model(model, X_test, y_test, model_name: str):
    """Print classification report and return metrics dict."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Phishing"]))

    auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC: {auc:.4f}")

    return {
        "auc": auc,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "report": classification_report(y_test, y_pred, output_dict=True)
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
    print(f"Saved confusion matrix → {save_path}")


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
    print(f"Saved ROC curves → {save_path}")


def train(csv_path: str, experiment_name: str = "url-phishing-detector"):
    """
    Full training pipeline with MLflow tracking.

    Args:
        csv_path: Path to dataset CSV with 'url' and 'label' columns
        experiment_name: MLflow experiment name
    """
    mlflow.set_experiment(experiment_name)

    X, y = load_dataset(csv_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

    pipelines = build_pipelines()
    results = {}
    best_model = None
    best_auc = 0.0

    for name, pipeline in pipelines.items():
        with mlflow.start_run(run_name=name):
            print(f"\nTraining {name}...")
            pipeline.fit(X_train, y_train)

            metrics = evaluate_model(pipeline, X_test, y_test, name)
            results[name] = metrics

            # Log to MLflow
            mlflow.log_metric("auc", metrics["auc"])
            mlflow.log_metric("f1_phishing", metrics["report"]["Phishing"]["f1-score"])
            mlflow.log_metric("precision_phishing", metrics["report"]["Phishing"]["precision"])
            mlflow.log_metric("recall_phishing", metrics["report"]["Phishing"]["recall"])
            mlflow.sklearn.log_model(pipeline, name)

            # Save confusion matrix
            cm_path = os.path.join(MODELS_DIR, f"cm_{name}.png")
            plot_confusion_matrix(y_test, metrics["y_pred"], name, cm_path)
            mlflow.log_artifact(cm_path)

            if metrics["auc"] > best_auc:
                best_auc = metrics["auc"]
                best_model = pipeline
                best_name = name

    # Save best model
    best_path = os.path.join(MODELS_DIR, "url_detector_best.joblib")
    joblib.dump(best_model, best_path)
    print(f"\n✅ Best model: {best_name} (AUC={best_auc:.4f}) saved → {best_path}")

    # ROC curve comparison
    roc_path = os.path.join(MODELS_DIR, "roc_curves_url.png")
    plot_roc_curves(results, y_test, roc_path)

    return best_model, results


if __name__ == "__main__":
    import sys
    csv = sys.argv[1] if len(sys.argv) > 1 else "data/processed/url_dataset.csv"
    train(csv)

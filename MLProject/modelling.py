import os
os.environ["MLFLOW_SERIALIZATION_FORMAT"] = "pickle"

import argparse
import warnings
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
if tracking_uri:
    mlflow.set_tracking_uri(tracking_uri)

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]
TARGET = "target"


def plot_confusion_matrix(y_true, y_pred, path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(model, feature_names, path):
    importances = model.feature_importances_
    idx = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(np.array(feature_names)[idx], importances[idx], color="steelblue")
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance (Random Forest)")
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def train(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features):
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "heart_disease_preprocessing.csv")
    df = pd.read_csv(data_path)
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth) if max_depth != "None" else None,
        min_samples_split=int(min_samples_split),
        min_samples_leaf=int(min_samples_leaf),
        max_features=str(max_features),
        random_state=42,
        n_jobs=-1,
    )

    cv_scores = cross_val_score(
        model, X_train, y_train, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="f1",
    )

    mlflow.set_experiment("heart-disease-ci")
    with mlflow.start_run(run_name="rf-mlproject-run") as run:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]


        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", int(n_estimators))
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", int(min_samples_split))
        mlflow.log_param("min_samples_leaf", int(min_samples_leaf))
        mlflow.log_param("max_features", str(max_features))
        mlflow.log_param("cv_folds", 5)

        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred, zero_division=0))
        mlflow.log_metric("recall", recall_score(y_test, y_pred, zero_division=0))
        mlflow.log_metric("f1", f1_score(y_test, y_pred, zero_division=0))
        mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_proba))
        mlflow.log_metric("cv_f1_mean", cv_scores.mean())
        mlflow.log_metric("cv_f1_std", cv_scores.std())

        mlflow.set_tag("mlproject", "heart-disease")
        mlflow.set_tag("ci", "github-actions")

        # Artefak
        artifact_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
        os.makedirs(artifact_dir, exist_ok=True)
        cm_path = os.path.join(artifact_dir, "confusion_matrix.png")
        fi_path = os.path.join(artifact_dir, "feature_importance.png")
        cr_path = os.path.join(artifact_dir, "classification_report.txt")

        plot_confusion_matrix(y_test, y_pred, cm_path)
        plot_feature_importance(model, FEATURES, fi_path)
        with open(cr_path, "w") as f:
            f.write(classification_report(y_test, y_pred, zero_division=0))

        mlflow.log_artifact(cm_path, artifact_path="artifacts")
        mlflow.log_artifact(fi_path, artifact_path="artifacts")
        mlflow.log_artifact(cr_path, artifact_path="artifacts")

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=X_train.head(3),
            signature=mlflow.models.infer_signature(X_train, model.predict(X_train)),
            serialization_format="pickle",
        )

        print(f"\n=== MLflow Project Run ===")
        print(f"Run ID: {run.info.run_id}")
        print(f"CV F1 Mean: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
        print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"Test F1: {f1_score(y_test, y_pred, zero_division=0):.4f}")
        print(f"Test ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=str, default="100")
    parser.add_argument("--max_depth", type=str, default="10")
    parser.add_argument("--min_samples_split", type=str, default="3")
    parser.add_argument("--min_samples_leaf", type=str, default="2")
    parser.add_argument("--max_features", type=str, default="sqrt")
    args = parser.parse_args()

    train(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
    )